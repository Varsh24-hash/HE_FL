import numpy as np
import tenseal as ts
import utils
import io
from azure.storage.blob import BlobServiceClient
import base64
# --- CONFIGURATION ---
HOSPITAL_ID = 2  # Change this to 2, 3, etc. for each PC
DOMAIN = 'mental'
AZURE_CONN_STR = "DefaultEndpointsProtocol=https;AccountName=hospitals;AccountKey="xxxxxx";EndpointSuffix=core.windows.net"
CONTAINER_NAME = "encrypted-hospitals"
print(f"--- Hospital {HOSPITAL_ID} Client ---")
blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONN_STR)
container_client = blob_service_client.get_container_client(CONTAINER_NAME)
# 1. SETUP ENCRYPTION
# We assume 'public.txt' and 'secret.txt' are already generated and present on this PC.
# (In a real setup, keys are distributed securely beforehand).
try:
    context_blob = container_client.get_blob_client("sending_files/public.txt")
    print(context_blob)
    exists = context_blob.exists()
    print("Exists:", exists)
    context_data = context_blob.download_blob().readall()
    context_data = base64.b64decode(context_data)
    context = ts.context_from(context_data)
    print("Loaded Public Key.")
except:
    # If this is the FIRST PC, generate keys and save them
    print("Key not found. Generating new keys (Lead Hospital Mode)...")
    exit()
    context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree = 16384, coeff_mod_bit_sizes = [31, 60, 60, 60, 60, 60, 60, 31])
    context.generate_galois_keys()
    context.global_scale = 2**60

    utils.write_data('./secret.txt', context.serialize(save_secret_key=True))
    context.make_context_public()
    #utils.write_data('./public.txt', context.serialize())
    print("New Keys Generated. SHARE 'public.txt' with other hospitals!")
# 2. ENCRYPT LOCAL WEIGHTS
weight_file = f'hospital_{HOSPITAL_ID}_{DOMAIN}.npy'
weights = np.load(weight_file)
enc_vec = ts.ckks_vector(context, weights)
# Save locally first
local_enc_file = f'arriving_files/encrypted_{DOMAIN}_{HOSPITAL_ID}.txt'
#utils.write_data(local_enc_file, enc_vec.serialize())
print(f"Weights Encrypted: {local_enc_file}")
serial = enc_vec.serialize()
# 3. UPLOAD TO AZURE
print("Uploading to Azure Blob Storage...")
blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=local_enc_file)
out_client = container_client.get_blob_client(local_enc_file)
out_client.upload_blob(serial, overwrite=True)
print("Upload Complete!")

