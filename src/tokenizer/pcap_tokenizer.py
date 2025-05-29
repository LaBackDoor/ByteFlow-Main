import logging
import os
import struct
from collections import defaultdict

from scapy.all import rdpcap
from scapy.layers.l2 import Ether
from scapy.layers.inet import IP, TCP, UDP, ICMP
from scapy.layers.inet6 import IPv6
from scapy.layers.dns import DNS
from scapy.layers.http import HTTP


class PCAPTokenizer:
    def __init__(self, vocab_size=280, offset=3):
        self.vocab_size = vocab_size
        self.offset = offset
        self.special_tokens = {
            'packet_start': 0x100 + offset,
            'packet_end': 0x101 + offset,  # Separate token for packet end
            'flow_end': 0x102 + offset,  # Separate token for flow end
            'field_sep': 0x103 + offset,  # Field separator token
        }
        if vocab_size < 261:  # Updated minimum size to account for the new tokens
            raise ValueError(f"Vocab size {vocab_size} is too small. Minimum is 261.")
        self.hex_to_token = {i: i + offset for i in range(256)}
        self.allocated_tokens = set(range(offset, 256 + offset))
        self.allocated_tokens.update(self.special_tokens.values())
        self.link_types = {
            0: self._allocate_token(), 1: self._allocate_token(),
            8: self._allocate_token(), 9: self._allocate_token(),
            10: self._allocate_token(), 101: self._allocate_token(),
            105: self._allocate_token(), 113: self._allocate_token(),
            127: self._allocate_token(),
        }
        self.flows = defaultdict(list)
        self.logger = logging.getLogger('PCAPTokenizer')

    def _allocate_token(self):
        for token_id in range(256 + self.offset, self.vocab_size + self.offset):
            if token_id not in self.allocated_tokens:
                self.allocated_tokens.add(token_id)
                return token_id
        raise ValueError(f"Token vocabulary limit of {self.vocab_size} exceeded")

    def tokenize_pcap(self, pcap_file):
        """
        Tokenize a PCAP file.
        Args:
            pcap_file: Path to the PCAP file.
        Returns:
            Dictionary mapping flow identifiers to token lists
        """
        try:
            packets_from_file = rdpcap(pcap_file)
        except Exception as e:
            self.logger.error(f"Error reading PCAP file '{pcap_file}': {e}")
            return {}

        if not packets_from_file:
            self.logger.warning(f"No packets found in PCAP file '{pcap_file}'.")
            return {}

        self.flows = defaultdict(list)  # Reset flows for this tokenization run

        base_name = os.path.basename(pcap_file)
        flow_id = f"{base_name}"

        sorted_packets = sorted(packets_from_file, key=lambda p: float(p.time))
        self.flows[flow_id] = sorted_packets

        # Tokenize each flow
        tokenized_flows_output = {}
        for flow_id, flow_packets_list in self.flows.items():
            if not flow_packets_list:
                self.logger.warning(f"Flow {flow_id} has no packets after extraction. Skipping.")
                continue
            tokenized_flows_output[flow_id] = self._tokenize_flow(flow_packets_list)

        return tokenized_flows_output

    def _tokenize_flow(self, packets_in_flow):
        tokens = []
        prev_time = None
        for packet in packets_in_flow:
            tokens.append(self.special_tokens['packet_start'])

            # Add link type with field separator
            link_type_token = self._get_link_type_token(packet)
            tokens.append(link_type_token)
            tokens.append(self.special_tokens['field_sep'])

            # Add timing information with field separator
            curr_time = float(packet.time)
            time_interval = curr_time - prev_time if prev_time is not None else 0.0
            if time_interval < 0:
                self.logger.warning(f"Negative time interval ({time_interval}s) detected. "
                                    "Ensure packets in flows are chronologically sorted.")
                # Optionally, clamp to 0 or handle as an error
                time_interval = 0.0
            time_tokens = self._encode_time_interval(time_interval)
            tokens.extend(time_tokens)
            tokens.append(self.special_tokens['field_sep'])

            prev_time = curr_time

            # Add packet data
            raw_data = bytes(packet)
            packet_tokens = self._encode_packet_data(raw_data, packet)
            tokens.extend(packet_tokens)

            # Add packet end token
            tokens.append(self.special_tokens['packet_end'])

        # Add flow end token
        tokens.append(self.special_tokens['flow_end'])
        return tokens

    def _get_link_type_token(self, packet):
        link_type = None
        if Ether in packet:
            link_type = 1
        elif hasattr(packet, 'linktype'):
            link_type = packet.linktype
        if link_type in self.link_types: return self.link_types[link_type]
        if link_type is not None:
            try:
                self.link_types[link_type] = self._allocate_token()
                return self.link_types[link_type]
            except ValueError:
                self.logger.warning(f"Vocab limit for link type {link_type}, using default.")
        return self.link_types.get(1, self._allocate_token())  # Default to Ethernet

    def _encode_time_interval(self, time_interval):
        packed = struct.pack('!d', time_interval)
        tokens = []
        # Tokenize each byte of the time interval with field separators
        for byte in packed:
            tokens.append(self.hex_to_token[byte])
            tokens.append(self.special_tokens['field_sep'])
        # Remove the last field separator
        if tokens and tokens[-1] == self.special_tokens['field_sep']:
            tokens.pop()
        return tokens

    def _encode_packet_data(self, raw_data, packet):
        """
        Encode packet data with field separators between individual fields within headers.
        This method identifies protocol headers and adds field separators between each field.

        Args:
            raw_data: Raw packet bytes
            packet: Scapy packet object

        Returns:
            List of token IDs with field separators between fields
        """
        tokens = []
        try:
            current_pos = 0

            # Process Ethernet header (if present)
            if Ether in packet:
                # Destination MAC (6 bytes)
                tokens.extend([self.hex_to_token[byte] for byte in raw_data[current_pos:current_pos + 6]])
                tokens.append(self.special_tokens['field_sep'])
                current_pos += 6

                # Source MAC (6 bytes)
                tokens.extend([self.hex_to_token[byte] for byte in raw_data[current_pos:current_pos + 6]])
                tokens.append(self.special_tokens['field_sep'])
                current_pos += 6

                # EtherType (2 bytes)
                tokens.extend([self.hex_to_token[byte] for byte in raw_data[current_pos:current_pos + 2]])
                tokens.append(self.special_tokens['field_sep'])
                current_pos += 2

            # Process IP header (if present)
            if IP in packet:
                ip_hdr = packet[IP]

                # Version & IHL (1 byte)
                tokens.append(self.hex_to_token[raw_data[current_pos]])
                tokens.append(self.special_tokens['field_sep'])
                current_pos += 1

                # DSCP & ECN (1 byte)
                tokens.append(self.hex_to_token[raw_data[current_pos]])
                tokens.append(self.special_tokens['field_sep'])
                current_pos += 1

                # Total Length (2 bytes)
                tokens.extend([self.hex_to_token[byte] for byte in raw_data[current_pos:current_pos + 2]])
                tokens.append(self.special_tokens['field_sep'])
                current_pos += 2

                # Identification (2 bytes)
                tokens.extend([self.hex_to_token[byte] for byte in raw_data[current_pos:current_pos + 2]])
                tokens.append(self.special_tokens['field_sep'])
                current_pos += 2

                # Flags & Fragment Offset (2 bytes)
                tokens.extend([self.hex_to_token[byte] for byte in raw_data[current_pos:current_pos + 2]])
                tokens.append(self.special_tokens['field_sep'])
                current_pos += 2

                # TTL (1 byte)
                tokens.append(self.hex_to_token[raw_data[current_pos]])
                tokens.append(self.special_tokens['field_sep'])
                current_pos += 1

                # Protocol (1 byte)
                tokens.append(self.hex_to_token[raw_data[current_pos]])
                tokens.append(self.special_tokens['field_sep'])
                current_pos += 1

                # Header Checksum (2 bytes)
                tokens.extend([self.hex_to_token[byte] for byte in raw_data[current_pos:current_pos + 2]])
                tokens.append(self.special_tokens['field_sep'])
                current_pos += 2

                # Source IP (4 bytes)
                tokens.extend([self.hex_to_token[byte] for byte in raw_data[current_pos:current_pos + 4]])
                tokens.append(self.special_tokens['field_sep'])
                current_pos += 4

                # Destination IP (4 bytes)
                tokens.extend([self.hex_to_token[byte] for byte in raw_data[current_pos:current_pos + 4]])
                tokens.append(self.special_tokens['field_sep'])
                current_pos += 4

                # Options (if any)
                ip_options_len = ip_hdr.ihl * 4 - 20  # Standard header is 20 bytes
                if ip_options_len > 0:
                    tokens.extend(
                        [self.hex_to_token[byte] for byte in raw_data[current_pos:current_pos + ip_options_len]])
                    tokens.append(self.special_tokens['field_sep'])
                    current_pos += ip_options_len

                # Process transport layer protocols
                if TCP in packet:
                    tcp_hdr = packet[TCP]

                    # Source Port (2 bytes)
                    tokens.extend([self.hex_to_token[byte] for byte in raw_data[current_pos:current_pos + 2]])
                    tokens.append(self.special_tokens['field_sep'])
                    current_pos += 2

                    # Destination Port (2 bytes)
                    tokens.extend([self.hex_to_token[byte] for byte in raw_data[current_pos:current_pos + 2]])
                    tokens.append(self.special_tokens['field_sep'])
                    current_pos += 2

                    # Sequence Number (4 bytes)
                    tokens.extend([self.hex_to_token[byte] for byte in raw_data[current_pos:current_pos + 4]])
                    tokens.append(self.special_tokens['field_sep'])
                    current_pos += 4

                    # Acknowledgment Number (4 bytes)
                    tokens.extend([self.hex_to_token[byte] for byte in raw_data[current_pos:current_pos + 4]])
                    tokens.append(self.special_tokens['field_sep'])
                    current_pos += 4

                    # Data Offset, Reserved, Flags (2 bytes)
                    tokens.extend([self.hex_to_token[byte] for byte in raw_data[current_pos:current_pos + 2]])
                    tokens.append(self.special_tokens['field_sep'])
                    current_pos += 2

                    # Window Size (2 bytes)
                    tokens.extend([self.hex_to_token[byte] for byte in raw_data[current_pos:current_pos + 2]])
                    tokens.append(self.special_tokens['field_sep'])
                    current_pos += 2

                    # Checksum (2 bytes)
                    tokens.extend([self.hex_to_token[byte] for byte in raw_data[current_pos:current_pos + 2]])
                    tokens.append(self.special_tokens['field_sep'])
                    current_pos += 2

                    # Urgent Pointer (2 bytes)
                    tokens.extend([self.hex_to_token[byte] for byte in raw_data[current_pos:current_pos + 2]])
                    tokens.append(self.special_tokens['field_sep'])
                    current_pos += 2

                    # TCP Options (if any)
                    tcp_options_len = tcp_hdr.dataofs * 4 - 20  # Standard header is 20 bytes
                    if tcp_options_len > 0:
                        tokens.extend(
                            [self.hex_to_token[byte] for byte in raw_data[current_pos:current_pos + tcp_options_len]])
                        tokens.append(self.special_tokens['field_sep'])
                        current_pos += tcp_options_len

                    # Try to identify application layer protocol
                    if DNS in packet:
                        self._encode_dns_header(raw_data, current_pos, tokens)
                    elif HTTP in packet:
                        self._encode_http_header(raw_data, current_pos, tokens)

                elif UDP in packet:
                    # Source Port (2 bytes)
                    tokens.extend([self.hex_to_token[byte] for byte in raw_data[current_pos:current_pos + 2]])
                    tokens.append(self.special_tokens['field_sep'])
                    current_pos += 2

                    # Destination Port (2 bytes)
                    tokens.extend([self.hex_to_token[byte] for byte in raw_data[current_pos:current_pos + 2]])
                    tokens.append(self.special_tokens['field_sep'])
                    current_pos += 2

                    # Length (2 bytes)
                    tokens.extend([self.hex_to_token[byte] for byte in raw_data[current_pos:current_pos + 2]])
                    tokens.append(self.special_tokens['field_sep'])
                    current_pos += 2

                    # Checksum (2 bytes)
                    tokens.extend([self.hex_to_token[byte] for byte in raw_data[current_pos:current_pos + 2]])
                    tokens.append(self.special_tokens['field_sep'])
                    current_pos += 2

                    # Try to identify application layer protocol
                    if DNS in packet:
                        self._encode_dns_header(raw_data, current_pos, tokens)

                elif ICMP in packet:
                    # Type (1 byte)
                    tokens.append(self.hex_to_token[raw_data[current_pos]])
                    tokens.append(self.special_tokens['field_sep'])
                    current_pos += 1

                    # Code (1 byte)
                    tokens.append(self.hex_to_token[raw_data[current_pos]])
                    tokens.append(self.special_tokens['field_sep'])
                    current_pos += 1

                    # Checksum (2 bytes)
                    tokens.extend([self.hex_to_token[byte] for byte in raw_data[current_pos:current_pos + 2]])
                    tokens.append(self.special_tokens['field_sep'])
                    current_pos += 2

                    # Rest of Header (4 bytes) - varies based on type and code
                    tokens.extend([self.hex_to_token[byte] for byte in raw_data[current_pos:current_pos + 4]])
                    tokens.append(self.special_tokens['field_sep'])
                    current_pos += 4

            # IPv6 handling
            elif IPv6 in packet:
                # Version, Traffic Class, Flow Label (4 bytes)
                tokens.extend([self.hex_to_token[byte] for byte in raw_data[current_pos:current_pos + 4]])
                tokens.append(self.special_tokens['field_sep'])
                current_pos += 4

                # Payload Length (2 bytes)
                tokens.extend([self.hex_to_token[byte] for byte in raw_data[current_pos:current_pos + 2]])
                tokens.append(self.special_tokens['field_sep'])
                current_pos += 2

                # Next Header (1 byte)
                tokens.append(self.hex_to_token[raw_data[current_pos]])
                tokens.append(self.special_tokens['field_sep'])
                current_pos += 1

                # Hop Limit (1 byte)
                tokens.append(self.hex_to_token[raw_data[current_pos]])
                tokens.append(self.special_tokens['field_sep'])
                current_pos += 1

                # Source Address (16 bytes)
                tokens.extend([self.hex_to_token[byte] for byte in raw_data[current_pos:current_pos + 16]])
                tokens.append(self.special_tokens['field_sep'])
                current_pos += 16

                # Destination Address (16 bytes)
                tokens.extend([self.hex_to_token[byte] for byte in raw_data[current_pos:current_pos + 16]])
                tokens.append(self.special_tokens['field_sep'])
                current_pos += 16

                # Similar transport layer handling as IPv4
                if TCP in packet or UDP in packet or ICMP in packet:
                    # We've already processed the IPv6 header, so we can reuse the same code
                    # for processing transport protocols
                    transport_tokens = self._encode_transport_layer(raw_data[current_pos:], packet)
                    tokens.extend(transport_tokens)
                    # Skip the rest of the packet since it's been processed
                    current_pos = len(raw_data)

            # Add remaining payload data if any
            if current_pos < len(raw_data):
                # Mark the payload section
                tokens.append(self.special_tokens['field_sep'])
                # Add payload data with separators every 16 bytes for structure
                payload = raw_data[current_pos:]
                for i in range(0, len(payload), 16):
                    chunk = payload[i:i + 16]
                    tokens.extend([self.hex_to_token[byte] for byte in chunk])
                    if i + 16 < len(payload):  # Don't add separator after the last chunk
                        tokens.append(self.special_tokens['field_sep'])

        except Exception as e:
            # Fall back to basic tokenization if parsing fails
            self.logger.warning(f"Packet parsing failed, falling back to basic tokenization: {e}")
            tokens = []
            # Add each byte with field separators every 16 bytes
            for i in range(0, len(raw_data), 16):
                chunk = raw_data[i:i + 16]
                tokens.extend([self.hex_to_token[byte] for byte in chunk])
                if i + 16 < len(raw_data):  # Don't add separator after the last chunk
                    tokens.append(self.special_tokens['field_sep'])

        # Remove the last field separator if it exists
        if tokens and tokens[-1] == self.special_tokens['field_sep']:
            tokens.pop()

        return tokens

    def _encode_dns_header(self, raw_data, current_pos, tokens):
        """Helper method to tokenize DNS header fields"""
        try:
            # Transaction ID (2 bytes)
            tokens.extend([self.hex_to_token[byte] for byte in raw_data[current_pos:current_pos + 2]])
            tokens.append(self.special_tokens['field_sep'])
            current_pos += 2

            # Flags (2 bytes)
            tokens.extend([self.hex_to_token[byte] for byte in raw_data[current_pos:current_pos + 2]])
            tokens.append(self.special_tokens['field_sep'])
            current_pos += 2

            # Questions (2 bytes)
            tokens.extend([self.hex_to_token[byte] for byte in raw_data[current_pos:current_pos + 2]])
            tokens.append(self.special_tokens['field_sep'])
            current_pos += 2

            # Answer RRs (2 bytes)
            tokens.extend([self.hex_to_token[byte] for byte in raw_data[current_pos:current_pos + 2]])
            tokens.append(self.special_tokens['field_sep'])
            current_pos += 2

            # Authority RRs (2 bytes)
            tokens.extend([self.hex_to_token[byte] for byte in raw_data[current_pos:current_pos + 2]])
            tokens.append(self.special_tokens['field_sep'])
            current_pos += 2

            # Additional RRs (2 bytes)
            tokens.extend([self.hex_to_token[byte] for byte in raw_data[current_pos:current_pos + 2]])
            tokens.append(self.special_tokens['field_sep'])
            current_pos += 2

            # We'll just mark the DNS queries/responses section
            if current_pos < len(raw_data):
                tokens.extend([self.hex_to_token[byte] for byte in raw_data[current_pos:]])

            return current_pos
        except Exception as e:
            self.logger.warning(f"Error parsing DNS header: {e}")
            return current_pos

    def _encode_http_header(self, raw_data, current_pos, tokens):
        """Helper method to tokenize HTTP header fields"""
        try:
            # This is a simple implementation since HTTP is text-based
            # We'll add field separators after each line
            http_data = raw_data[current_pos:]

            # Try to decode as ASCII (most HTTP traffic is ASCII)
            try:
                http_text = http_data.decode('ascii', errors='ignore')
                lines = http_text.split('\r\n')

                for i, line in enumerate(lines):
                    if not line:  # Empty line marks end of headers
                        if i < len(lines) - 1:  # There's content after headers
                            tokens.append(self.special_tokens['field_sep'])

                    line_bytes = line.encode('ascii', errors='ignore')
                    tokens.extend([self.hex_to_token[byte] for byte in line_bytes])

                    if i < len(lines) - 1:  # Don't add after the last line
                        tokens.append(self.special_tokens['field_sep'])
            except:
                # Fallback if decoding fails
                for i in range(0, len(http_data), 16):
                    chunk = http_data[i:i + 16]
                    tokens.extend([self.hex_to_token[byte] for byte in chunk])
                    if i + 16 < len(http_data):
                        tokens.append(self.special_tokens['field_sep'])

            return current_pos + len(http_data)
        except Exception as e:
            self.logger.warning(f"Error parsing HTTP header: {e}")
            return current_pos

    def _encode_transport_layer(self, data, packet):
        """Helper method to encode transport layer headers"""
        tokens = []
        current_pos = 0

        try:
            if TCP in packet:
                tcp_hdr = packet[TCP]

                # Source Port (2 bytes)
                tokens.extend([self.hex_to_token[byte] for byte in data[current_pos:current_pos + 2]])
                tokens.append(self.special_tokens['field_sep'])
                current_pos += 2

                # Destination Port (2 bytes)
                tokens.extend([self.hex_to_token[byte] for byte in data[current_pos:current_pos + 2]])
                tokens.append(self.special_tokens['field_sep'])
                current_pos += 2

                # Sequence Number (4 bytes)
                tokens.extend([self.hex_to_token[byte] for byte in data[current_pos:current_pos + 4]])
                tokens.append(self.special_tokens['field_sep'])
                current_pos += 4

                # Acknowledgment Number (4 bytes)
                tokens.extend([self.hex_to_token[byte] for byte in data[current_pos:current_pos + 4]])
                tokens.append(self.special_tokens['field_sep'])
                current_pos += 4

                # Data Offset, Reserved, Flags (2 bytes)
                tokens.extend([self.hex_to_token[byte] for byte in data[current_pos:current_pos + 2]])
                tokens.append(self.special_tokens['field_sep'])
                current_pos += 2

                # Window Size (2 bytes)
                tokens.extend([self.hex_to_token[byte] for byte in data[current_pos:current_pos + 2]])
                tokens.append(self.special_tokens['field_sep'])
                current_pos += 2

                # Checksum (2 bytes)
                tokens.extend([self.hex_to_token[byte] for byte in data[current_pos:current_pos + 2]])
                tokens.append(self.special_tokens['field_sep'])
                current_pos += 2

                # Urgent Pointer (2 bytes)
                tokens.extend([self.hex_to_token[byte] for byte in data[current_pos:current_pos + 2]])
                tokens.append(self.special_tokens['field_sep'])
                current_pos += 2

                # TCP Options (if any)
                tcp_options_len = tcp_hdr.dataofs * 4 - 20  # Standard header is 20 bytes
                if tcp_options_len > 0:
                    tokens.extend([self.hex_to_token[byte] for byte in data[current_pos:current_pos + tcp_options_len]])
                    tokens.append(self.special_tokens['field_sep'])
                    current_pos += tcp_options_len

            elif UDP in packet:
                # Source Port (2 bytes)
                tokens.extend([self.hex_to_token[byte] for byte in data[current_pos:current_pos + 2]])
                tokens.append(self.special_tokens['field_sep'])
                current_pos += 2

                # Destination Port (2 bytes)
                tokens.extend([self.hex_to_token[byte] for byte in data[current_pos:current_pos + 2]])
                tokens.append(self.special_tokens['field_sep'])
                current_pos += 2

                # Length (2 bytes)
                tokens.extend([self.hex_to_token[byte] for byte in data[current_pos:current_pos + 2]])
                tokens.append(self.special_tokens['field_sep'])
                current_pos += 2

                # Checksum (2 bytes)
                tokens.extend([self.hex_to_token[byte] for byte in data[current_pos:current_pos + 2]])
                tokens.append(self.special_tokens['field_sep'])
                current_pos += 2

            elif ICMP in packet:
                # Type (1 byte)
                tokens.append(self.hex_to_token[data[current_pos]])
                tokens.append(self.special_tokens['field_sep'])
                current_pos += 1

                # Code (1 byte)
                tokens.append(self.hex_to_token[data[current_pos]])
                tokens.append(self.special_tokens['field_sep'])
                current_pos += 1

                # Checksum (2 bytes)
                tokens.extend([self.hex_to_token[byte] for byte in data[current_pos:current_pos + 2]])
                tokens.append(self.special_tokens['field_sep'])
                current_pos += 2

                # Rest of Header (4 bytes) - varies based on type and code
                tokens.extend([self.hex_to_token[byte] for byte in data[current_pos:current_pos + 4]])
                tokens.append(self.special_tokens['field_sep'])
                current_pos += 4

            # Add remaining data
            if current_pos < len(data):
                tokens.extend([self.hex_to_token[byte] for byte in data[current_pos:]])

        except Exception as e:
            self.logger.warning(f"Error encoding transport layer: {e}")
            # Add remaining data if there was an error
            if current_pos < len(data):
                tokens.extend([self.hex_to_token[byte] for byte in data[current_pos:]])

        return tokens

    def decode_flow(self, tokens):
        token_to_link_type = {v: k for k, v in self.link_types.items()}
        packets_info = []
        i = 0

        while i < len(tokens):
            if tokens[i] == self.special_tokens['packet_start']:
                i += 1
                if i >= len(tokens):
                    self.logger.warning("Unexpected end after packet start")
                    break

                # Parse link type
                link_type_token = tokens[i]
                link_type = token_to_link_type.get(link_type_token, 1)
                i += 1

                # Skip field separator after link type
                if i < len(tokens) and tokens[i] == self.special_tokens['field_sep']:
                    i += 1
                else:
                    self.logger.warning(f"Expected field separator at position {i}, continuing anyway")

                # Parse time interval
                if i + 8 > len(tokens):
                    self.logger.warning("Not enough tokens for time")
                    break

                # Skip field separators between time bytes
                time_tokens = []
                for _ in range(8):  # Expect 8 bytes for time
                    if i >= len(tokens):
                        break
                    if tokens[i] == self.special_tokens['field_sep']:
                        i += 1
                        continue
                    time_tokens.append(tokens[i])
                    i += 1
                    # Skip field separator if present
                    if i < len(tokens) and tokens[i] == self.special_tokens['field_sep']:
                        i += 1

                # If we didn't get 8 time tokens, something went wrong
                if len(time_tokens) != 8:
                    self.logger.warning(f"Expected 8 time tokens, got {len(time_tokens)}")
                    # Try to continue anyway
                    time_tokens.extend([0] * (8 - len(time_tokens)))

                # Convert time tokens to time value
                time_bytes = bytes(
                    [(t - self.offset) if self.offset <= t < (256 + self.offset) else 0 for t in time_tokens])
                time_interval = struct.unpack('!d', time_bytes)[0]

                # Parse packet data, skipping internal field separators
                packet_data_bytes = []
                while (i < len(tokens) and
                       tokens[i] != self.special_tokens['packet_start'] and
                       tokens[i] != self.special_tokens['packet_end'] and
                       tokens[i] != self.special_tokens['flow_end']):

                    if tokens[i] == self.special_tokens['field_sep']:
                        # Skip field separators within packet data
                        i += 1
                        continue

                    byte_value = (tokens[i] - self.offset) if self.offset <= tokens[i] < (256 + self.offset) else 0
                    packet_data_bytes.append(byte_value)
                    i += 1

                # Add the packet info to our list
                packets_info.append((link_type, time_interval, bytes(packet_data_bytes)))

                # Skip packet end token if present
                if i < len(tokens) and tokens[i] == self.special_tokens['packet_end']:
                    i += 1

            elif tokens[i] == self.special_tokens['flow_end']:
                # End of flow reached
                i += 1
                break
            elif tokens[i] == self.special_tokens['field_sep']:
                self.logger.warning(f"Unexpected field separator at position {i}")
                i += 1
            else:
                self.logger.warning(f"Unexpected token {tokens[i]} at pos {i}")
                i += 1

        return packets_info