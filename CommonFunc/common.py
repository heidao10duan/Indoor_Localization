import base64


def decode_base64(data):
    missing_padding = 4 - len(data) % 4
    if missing_padding:
        data += '='*missing_padding
    return base64.b64decode(data)