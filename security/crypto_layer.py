from cryptography.fernet import Fernet
import hashlib

# Generate key once
key = Fernet.generate_key()
cipher = Fernet(key)

def encrypt_data(data_string):
    return cipher.encrypt(data_string.encode())

def decrypt_data(encrypted_data):
    return cipher.decrypt(encrypted_data).decode()

def generate_hash(data_string):
    return hashlib.sha256(data_string.encode()).hexdigest()

def verify_hash(original_data, received_hash):
    return generate_hash(original_data) == received_hash