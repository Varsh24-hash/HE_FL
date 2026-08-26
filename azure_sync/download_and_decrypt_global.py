
import tenseal as ts
import numpy as np
import utils
from azure.storage.blob import BlobServiceClient
import os
# --- CONFIGURATION ---
# This matches the configuration used in encryption/aggregation
DOMAIN = 'genetic'
AZURE_CONN_STR = "Apoorva's connection string"
CONTAINER_NAME = "hospital-weights"
GLOBAL_MODEL_NAME = f"global_encrypted_{DOMAIN}.txt"
print(f"--- Hospital Client: Download & Decrypt Global {DOMAIN.upper()} Model ---")
try:
    # 1. Download Global Encrypted Model from Azure
    print("Connecting to Azure...")
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONN_STR)
    blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=GLOBAL_MODEL_NAME)

    print(f"Downloading {GLOBAL_MODEL_NAME}...")
    with open(GLOBAL_MODEL_NAME, "wb") as download_file:
        download_file.write(blob_client.download_blob().readall())
    print("Download complete.")
    # 2. Load Secret Key (Context)
    # The client needs their secret key to decrypt the result.
    # This assumes 'secret.txt' was generated during the encryption phase and is present locally.
    print("Loading Secret Context...")
    if not os.path.exists('./secret.txt'):
        raise FileNotFoundError("'secret.txt' not found. Cannot decrypt without the private key.")

    context = ts.context_from(utils.read_data('./secret.txt'))
    # 3. Load the Encrypted Vector
    print("Loading encrypted vector...")
    file_data = utils.read_data(GLOBAL_MODEL_NAME)
    enc_vec = ts.lazy_ckks_vector_from(file_data)
    enc_vec.link_context(context)
    # 4. Decrypt
    print("Decrypting weights...")
    # This uses the secret key loaded in the context to reveal the plain numbers
    global_weights_list = enc_vec.decrypt()
    global_weights_np = np.array(global_weights_list)
    print(f"Decryption Successful!")
    print(f"Global Weights Shape: {global_weights_np.shape}")
    print(f"First 5 weights: {global_weights_np[:5]}")
    # 5. Save Final Weights for Local Model Update
    output_filename = f'updated_global_weights_{DOMAIN}.npy'
    np.save(output_filename, global_weights_np)
    print(f"Saved updated weights to: {output_filename}")
    print("You can now load this .npy file into your local model using 'model.coefs_ = ...'")
except Exception as e:
    print(f"Error during download/decryption: {e}")
