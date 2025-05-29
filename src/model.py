import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RelativePositionBias(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_buckets = config.relative_attention_num_buckets
        self.max_distance = 128
        self.relative_attention_bias = nn.Embedding(self.num_buckets, config.num_heads)

    def compute_bias(self, q_abs_pos, k_abs_pos):
        relative_position = k_abs_pos.unsqueeze(1) - q_abs_pos.unsqueeze(2)

        relative_position_bucket = self._relative_position_bucket(
            relative_position,
            bidirectional=True,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance
        )

        values = self.relative_attention_bias(relative_position_bucket)
        values = values.permute(0, 3, 1, 2)

        return values

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        ret = 0
        n = -relative_position
        if bidirectional:
            num_buckets //= 2
            ret += (n < 0).long() * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n, device=n.device))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
                torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()

        val_if_large = torch.min(val_if_large, torch.full_like(n, num_buckets - 1, device=n.device))

        ret += torch.where(is_small, n, val_if_large)
        return ret

class RoutingNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads

        hidden_dim = config.d_model * 4

        self.up_proj = nn.Linear(config.d_model, hidden_dim)
        self.gate_proj = nn.Linear(config.d_model, hidden_dim)
        self.down_proj = nn.Linear(hidden_dim, config.num_heads * 2)

    def forward(self, hidden_states):
        batch_size, seq_len, _ = hidden_states.shape
        up_states = self.up_proj(hidden_states)
        gate_states = self.gate_proj(hidden_states)

        activated_states = F.silu(up_states) * gate_states

        logits = self.down_proj(activated_states)

        logits = logits.view(batch_size, seq_len, self.num_heads, 2)

        routing_weights = F.softmax(logits, dim=-1)

        entropy = -torch.sum(routing_weights * torch.log(routing_weights + 1e-9), dim=-1)
        mean_entropy_loss = torch.mean(entropy)

        return routing_weights.permute(0, 2, 1, 3), mean_entropy_loss


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.d_model = config.d_model
        self.d_head = config.d_model // config.num_heads

        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, config.d_model)
        self.v_proj = nn.Linear(config.d_model, config.d_model)
        self.o_proj = nn.Linear(config.d_model, config.d_model)

        self.dropout = nn.Dropout(config.dropout_rate)
        self.routing_network = RoutingNetwork(config)

    @staticmethod
    def _get_permutation_indices(major_pos, minor_pos, attention_mask_for_padding):
        batch_size, seq_len_orig = major_pos.shape
        device = major_pos.device

        all_perm_indices = []
        all_inv_perm_indices = []
        max_effective_len = 0

        for b in range(batch_size):
            valid_idx = torch.where(attention_mask_for_padding[b] == 1)[0]
            s_eff = len(valid_idx)
            if s_eff == 0:
                all_perm_indices.append(torch.empty(0, dtype=torch.long, device=device))
                all_inv_perm_indices.append(torch.empty(0, dtype=torch.long, device=device))
                continue

            major_pos_b_valid = major_pos[b][valid_idx]
            minor_pos_b_valid = minor_pos[b][valid_idx]
            max_minor_val = minor_pos.max() + 1

            composite_key = major_pos_b_valid * max_minor_val + minor_pos_b_valid

            perm_b_eff = torch.argsort(composite_key)
            perm_indices_orig_space = valid_idx[perm_b_eff]
            all_perm_indices.append(perm_indices_orig_space)

            inv_perm_b_eff = torch.empty(s_eff, dtype=torch.long, device=device)
            inv_perm_b_eff[perm_b_eff] = torch.arange(s_eff, device=device)
            all_inv_perm_indices.append(inv_perm_b_eff)

            if s_eff > max_effective_len:
                max_effective_len = s_eff

        perm_indices_batched = torch.zeros(batch_size, seq_len_orig, dtype=torch.long, device=device)
        inv_perm_indices_batched = torch.zeros(batch_size, seq_len_orig, dtype=torch.long, device=device)

        for b in range(batch_size):
            max_val = minor_pos[b].max().item() + 1
            composite_key = major_pos[b] * max_val + minor_pos[b]
            perm_b = torch.argsort(composite_key)
            perm_indices_batched[b] = perm_b

            inv_perm_b = torch.empty_like(perm_b)
            inv_perm_b[perm_b] = torch.arange(seq_len_orig, device=device)
            inv_perm_indices_batched[b] = inv_perm_b

        return perm_indices_batched, inv_perm_indices_batched

    @staticmethod
    def _permute_tensor(x, perm_indices):
        B, H, S, D = x.shape
        perm_expanded = perm_indices.unsqueeze(1).unsqueeze(-1).expand(B, H, S, D)
        return torch.gather(x, 2, perm_expanded)

    @staticmethod
    def _permute_bias(bias, perm_indices):
        B, H, S, _ = bias.shape
        idx_q = perm_indices.unsqueeze(1).unsqueeze(-1).expand(B, H, S, S)
        bias_perm_q = torch.gather(bias, 2, idx_q)
        idx_k = perm_indices.unsqueeze(1).unsqueeze(2).expand(B, H, S, S)
        bias_perm_k_q = torch.gather(bias_perm_q, 3, idx_k)
        return bias_perm_k_q

    @staticmethod
    def _permute_mask(attention_mask, perm_indices_k):
        if attention_mask is None:
            return None
        B, _, Q_dim_mask, K_orig = attention_mask.shape
        K_new = perm_indices_k.shape[1]

        perm_expanded = perm_indices_k.unsqueeze(1).unsqueeze(1).expand(B, 1, Q_dim_mask, K_new)
        return torch.gather(attention_mask, 3, perm_expanded)

    def forward(self, query, key, value,
                attention_padding_mask=None,
                causal_mask_for_row_view=None,
                position_bias_row=None,
                position_indices=None,
                relative_attention_bias_module=None,
                hidden_state_for_router=None):

        batch_size = query.size(0)
        q_orig_len = query.size(1)
        device = query.device

        q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)

        routing_weights, entropy_loss = self.routing_network(hidden_state_for_router)

        # Row-wise Attention
        scores_row = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if position_bias_row is not None:
            scores_row += position_bias_row

        final_row_mask = None
        if attention_padding_mask is not None:
            final_row_mask = attention_padding_mask
        if causal_mask_for_row_view is not None:
            final_row_mask = final_row_mask * causal_mask_for_row_view if final_row_mask is not None else causal_mask_for_row_view

        if final_row_mask is not None:
            scores_row = scores_row.masked_fill(final_row_mask == 0, -1e9)

        attn_weights_row = F.softmax(scores_row, dim=-1)
        attn_weights_row = self.dropout(attn_weights_row)
        context_row = torch.matmul(attn_weights_row, v)

        # Column-wise Attention
        if position_indices is not None and relative_attention_bias_module is not None:
            abs_col_pos = position_indices[..., 1]
            abs_row_pos = position_indices[..., 0]

            perm_indices, inv_perm_indices = self._get_permutation_indices(abs_col_pos, abs_row_pos,
                                                                           attention_padding_mask.squeeze(1).squeeze(
                                                                               1) if attention_padding_mask is not None else torch.ones_like(
                                                                               abs_col_pos))

            q_col_perm = self._permute_tensor(q, perm_indices)
            k_col_perm = self._permute_tensor(k, perm_indices)
            v_col_perm = self._permute_tensor(v, perm_indices)

            perm_abs_col_pos_q = torch.gather(abs_col_pos, 1, perm_indices)
            perm_abs_col_pos_k = perm_abs_col_pos_q

            position_bias_col_permuted = relative_attention_bias_module.compute_bias(
                perm_abs_col_pos_q, perm_abs_col_pos_k
            )

            causal_mask_for_permuted_view = torch.tril(torch.ones((q_orig_len, q_orig_len), device=device)).unsqueeze(
                0).unsqueeze(0)

            final_perm_col_mask = causal_mask_for_permuted_view
            if attention_padding_mask is not None:
                permuted_padding_mask_k = self._permute_mask(attention_padding_mask, perm_indices)
                final_perm_col_mask = final_perm_col_mask * permuted_padding_mask_k

            scores_col_perm = torch.matmul(q_col_perm, k_col_perm.transpose(-2, -1)) / math.sqrt(self.d_head)
            if position_bias_col_permuted is not None:
                scores_col_perm += position_bias_col_permuted

            if final_perm_col_mask is not None:
                scores_col_perm = scores_col_perm.masked_fill(final_perm_col_mask == 0, -1e9)

            attn_weights_col_perm = F.softmax(scores_col_perm, dim=-1)
            attn_weights_col_perm = self.dropout(attn_weights_col_perm)
            context_col_perm = torch.matmul(attn_weights_col_perm, v_col_perm)

            context_col = self._permute_tensor(context_col_perm, inv_perm_indices)
        else:
            scores_col = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
            if final_row_mask is not None:
                scores_col = scores_col.masked_fill(final_row_mask == 0, -1e9)
            attn_weights_col = F.softmax(scores_col, dim=-1)
            context_col = torch.matmul(self.dropout(attn_weights_col), v)

        context = routing_weights[..., 0].unsqueeze(-1) * context_row + \
                  routing_weights[..., 1].unsqueeze(-1) * context_col

        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.o_proj(context)

        return output, entropy_loss


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense1 = nn.Linear(config.d_model, config.d_ff)
        self.dense2 = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x):
        x = self.dense1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attention = MultiHeadAttention(config)
        self.layer_norm1 = nn.LayerNorm(config.d_model, eps=1e-6)
        self.feed_forward = FeedForward(config)
        self.layer_norm2 = nn.LayerNorm(config.d_model, eps=1e-6)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x, attention_padding_mask=None, causal_mask_for_row_view=None,
                position_bias_row=None,
                position_indices=None,
                relative_attention_bias_module=None):
        norm_x = self.layer_norm1(x)
        attn_output, entropy_loss = self.self_attention(
            norm_x, norm_x, norm_x,
            attention_padding_mask=attention_padding_mask,
            causal_mask_for_row_view=causal_mask_for_row_view,
            position_bias_row=position_bias_row,
            position_indices=position_indices,
            relative_attention_bias_module=relative_attention_bias_module,
            hidden_state_for_router=norm_x
        )
        x = x + self.dropout(attn_output)
        norm_x = self.layer_norm2(x)
        ff_output = self.feed_forward(norm_x)
        x = x + self.dropout(ff_output)
        return x, entropy_loss


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attention = MultiHeadAttention(config)
        self.layer_norm1 = nn.LayerNorm(config.d_model, eps=1e-6)
        self.cross_attention = MultiHeadAttention(config)
        self.layer_norm2 = nn.LayerNorm(config.d_model, eps=1e-6)
        self.feed_forward = FeedForward(config)
        self.layer_norm3 = nn.LayerNorm(config.d_model, eps=1e-6)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x, encoder_output,
                self_attention_padding_mask=None,
                self_causal_mask=None,
                self_pos_bias_row=None,
                self_position_indices=None,
                cross_attention_padding_mask=None,
                cross_pos_bias_row=None,
                encoder_position_indices_for_cross=None,
                decoder_position_indices_for_cross_q=None,
                relative_attention_bias_module=None):
        total_layer_entropy_loss = 0.0

        norm_x_self = self.layer_norm1(x)
        attn_output, sa_entropy = self.self_attention(
            norm_x_self, norm_x_self, norm_x_self,
            attention_padding_mask=self_attention_padding_mask,
            causal_mask_for_row_view=self_causal_mask,
            position_bias_row=self_pos_bias_row,
            position_indices=self_position_indices,
            relative_attention_bias_module=relative_attention_bias_module,
            hidden_state_for_router=norm_x_self
        )
        total_layer_entropy_loss += sa_entropy
        x = x + self.dropout(attn_output)

        norm_x_cross = self.layer_norm2(x)

        cross_attn_output, ca_entropy = self.cross_attention(
            query=norm_x_cross, key=encoder_output, value=encoder_output,
            attention_padding_mask=cross_attention_padding_mask,
            causal_mask_for_row_view=None,
            position_bias_row=cross_pos_bias_row,
            position_indices=None,
            relative_attention_bias_module=None,
            hidden_state_for_router=norm_x_cross
        )
        total_layer_entropy_loss += ca_entropy
        x = x + self.dropout(cross_attn_output)

        norm_x_ff = self.layer_norm3(x)
        ff_output = self.feed_forward(norm_x_ff)
        x = x + self.dropout(ff_output)
        return x, total_layer_entropy_loss


class ByT5Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.encoder_layers)])
        self.final_layer_norm = nn.LayerNorm(config.d_model, eps=1e-6)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.relative_attention_bias = RelativePositionBias(config) if config.use_relative_attention_bias else None

    def forward(self, input_ids, attention_mask=None, position_indices=None):
        total_entropy_loss = 0.0

        encoder_key_padding_mask = attention_mask.unsqueeze(1).unsqueeze(
            2) if attention_mask is not None else None

        causal_mask_for_row_view_encoder = None

        position_bias_row = None
        if self.relative_attention_bias is not None and position_indices is not None:
            abs_pos_row = position_indices[..., 0]
            position_bias_row = self.relative_attention_bias.compute_bias(abs_pos_row, abs_pos_row)

        x = self.embed(input_ids)
        x = self.dropout(x)

        for layer in self.layers:
            x, layer_entropy_loss = layer(
                x,
                attention_padding_mask=encoder_key_padding_mask,
                causal_mask_for_row_view=causal_mask_for_row_view_encoder,
                position_bias_row=position_bias_row,
                position_indices=position_indices,
                relative_attention_bias_module=self.relative_attention_bias
            )
            total_entropy_loss += layer_entropy_loss

        x = self.final_layer_norm(x)
        return x, total_entropy_loss


class ByT5Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.decoder_layers)])
        self.final_layer_norm = nn.LayerNorm(config.d_model, eps=1e-6)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.relative_attention_bias = RelativePositionBias(config) if config.use_relative_attention_bias else None

    def forward(self, input_ids, encoder_output,
                encoder_key_padding_mask=None,
                decoder_padding_mask=None,
                encoder_position_indices=None,
                decoder_position_indices=None
                ):
        batch_size, dec_seq_length = input_ids.size()
        device = input_ids.device
        total_entropy_loss = 0.0

        self_attn_key_padding_mask = decoder_padding_mask.unsqueeze(1).unsqueeze(
            2) if decoder_padding_mask is not None else None

        self_attn_causal_mask = torch.tril(torch.ones((dec_seq_length, dec_seq_length), device=device)).unsqueeze(
            0).unsqueeze(0)

        self_pos_bias_row = None
        if self.relative_attention_bias is not None and decoder_position_indices is not None:
            dec_abs_pos_row = decoder_position_indices[..., 0]
            self_pos_bias_row = self.relative_attention_bias.compute_bias(dec_abs_pos_row, dec_abs_pos_row)

        cross_attn_key_padding_mask = encoder_key_padding_mask.unsqueeze(1).unsqueeze(
            2) if encoder_key_padding_mask is not None else None  # [B,1,1,S_enc]

        cross_pos_bias_row = None
        if self.relative_attention_bias is not None and decoder_position_indices is not None and encoder_position_indices is not None:
            dec_abs_pos_row_for_q = decoder_position_indices[..., 0]
            enc_abs_pos_row_for_k = encoder_position_indices[..., 0]
            cross_pos_bias_row = self.relative_attention_bias.compute_bias(dec_abs_pos_row_for_q, enc_abs_pos_row_for_k)

        x = self.embed(input_ids)
        x = self.dropout(x)

        for layer in self.layers:
            x, layer_entropy_loss = layer(
                x, encoder_output,
                self_attention_padding_mask=self_attn_key_padding_mask,
                self_causal_mask=self_attn_causal_mask,
                self_pos_bias_row=self_pos_bias_row,
                self_position_indices=decoder_position_indices,
                cross_attention_padding_mask=cross_attn_key_padding_mask,
                cross_pos_bias_row=cross_pos_bias_row,
                encoder_position_indices_for_cross=encoder_position_indices,
                decoder_position_indices_for_cross_q=decoder_position_indices,
                relative_attention_bias_module=self.relative_attention_bias
            )
            total_entropy_loss += layer_entropy_loss

        x = self.final_layer_norm(x)
        return x, total_entropy_loss


class ByT5Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = ByT5Encoder(config)
        self.decoder = ByT5Decoder(config)

    def forward(self, encoder_input_ids, decoder_input_ids=None,
                encoder_attention_mask=None,
                decoder_attention_mask=None,
                encoder_position_indices=None,
                decoder_position_indices=None):
        encoder_output, encoder_entropy_loss = self.encoder(
            encoder_input_ids,
            attention_mask=encoder_attention_mask,
            position_indices=encoder_position_indices
        )

        total_entropy_loss = encoder_entropy_loss

        if decoder_input_ids is not None:
            decoder_output, decoder_entropy_loss = self.decoder(
                decoder_input_ids,
                encoder_output,
                encoder_key_padding_mask=encoder_attention_mask,
                decoder_padding_mask=decoder_attention_mask,
                encoder_position_indices=encoder_position_indices,
                decoder_position_indices=decoder_position_indices
            )
            total_entropy_loss += decoder_entropy_loss
            return encoder_output, decoder_output, total_entropy_loss

        return encoder_output, total_entropy_loss


class ByT5ForConditionalGeneration(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = ByT5Model(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.device = None

    def to(self, *args, **kwargs):
        device, *_ = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None:
            self.device = device
        return super().to(*args, **kwargs)

    def forward(self, encoder_input_ids, decoder_input_ids=None,
                encoder_attention_mask=None, decoder_attention_mask=None,
                encoder_position_indices=None,
                decoder_position_indices=None
                ):

        _, decoder_output, total_entropy_loss = self.model(
            encoder_input_ids, decoder_input_ids,
            encoder_attention_mask, decoder_attention_mask,
            encoder_position_indices, decoder_position_indices
        )

        lm_logits = self.lm_head(decoder_output)
        return lm_logits, total_entropy_loss

    def generate(self, input_ids, attention_mask=None, position_indices=None, max_length=20):
        batch_size = input_ids.size(0)
        device = input_ids.device

        encoder_output, encoder_entropy = self.model.encoder(
            input_ids,
            attention_mask=attention_mask,
            position_indices=position_indices
        )

        decoder_input_ids = torch.ones((batch_size, 1), dtype=torch.long, device=device) * self.config.eos_token_id

        generated_ids_list = [decoder_input_ids]

        for _ in range(max_length - 1):
            current_decoder_len = decoder_input_ids.size(1)

            dec_pos_row = torch.arange(current_decoder_len, device=device).unsqueeze(0).repeat(batch_size, 1)

            dec_pos_col = dec_pos_row.clone()
            current_decoder_position_indices = torch.stack((dec_pos_row, dec_pos_col), dim=-1)  # [B, current_len, 2]

            decoder_output, decoder_entropy = self.model.decoder(
                decoder_input_ids,
                encoder_output,
                encoder_position_indices=position_indices,
                decoder_position_indices=current_decoder_position_indices,
                cross_attention_mask=attention_mask
            )

            logits = self.lm_head(decoder_output[:, -1:, :])
            next_token = torch.argmax(logits, dim=-1)

            decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=1)
            generated_ids_list.append(next_token)

            if (next_token == self.config.eos_token_id).all():
                break

        return torch.cat(generated_ids_list, dim=1)