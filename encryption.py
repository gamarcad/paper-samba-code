# =======================================================================
# file: encryption.py
# description: Encryption utility.
# =======================================================================
from Crypto.Random import get_random_bytes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

# Encryption constants
AES_256_KEY_LENGTH_BYTES = 32


def generate_symetric_key():
    """Returns a new symmetric key."""
    return get_random_bytes(32)


class Cipher:
    """
    Cipher is an abstraction of a cipher object that allows to encrypt/decrypt an message.
    """
    def encrypt(self, plaintext): pass
    def decrypt(self, ciphertext): pass


class AES256GCMCipher(Cipher):
    """
    It uses the AES-256-GCM implementation available in the Ionic cryptographic SDK.
    https://dev.ionic.com/sdk/tasks/crypto-aes-gcm
    """
    def __init__(self, key):
        # AES-256-GCM requires a key length of 256 bits (32 bytes)
        if len(key) != AES_256_KEY_LENGTH_BYTES:
            raise Exception("Invalid AES-256-GCM key length: requires 256 bits length key")
        self.key = key
        self.cipher = AESGCM(self.key)

    def encrypt(self, plaintext):
        """Returns the plaintext encrypted with the symmetric key."""
        plaintext_type = type(plaintext)
        if plaintext_type not in [str, bytes]:
            plaintext = str(plaintext)
        if type(plaintext) == str:
            plaintext = plaintext.encode('utf-8')
        return self.__encrypt_aes_gcm(plaintext), plaintext_type

    def decrypt(self, ciphertext):
        """Returns the ciphertext decrypted with the symmetric key."""
        ciphertext, data_type = ciphertext
        plaintext: bytes = self.__decrypt_aes_gcm(ciphertext)
        if data_type == str:
            return plaintext.decode('utf-8')
        else:
            return data_type(plaintext)

    def __encrypt_aes_gcm(self, message : bytes):
        """Returns the given plaintext encrypted with AES-GCM."""
        nonce = get_random_bytes(12)
        return nonce, self.cipher.encrypt(
            nonce=nonce,
            data=message,
            associated_data=b'',
        )

    def __decrypt_aes_gcm(self, encrypted_data : (bytes,bytes)) -> bytes:
        """Returns the decrypted ciphertext with AES-GCM."""
        nonce, encrypted_data = encrypted_data
        return self.cipher.decrypt(
            nonce=nonce,
            data=encrypted_data,
            associated_data=b''
        )

