#include "openfhe.h"
#include "ciphertext-ser.h"
#include "cryptocontext-ser.h"
#include "key/key-ser.h"
#include "scheme/ckksrns/ckksrns-ser.h"
#include <iostream>
#include <vector>
using namespace lbcrypto;
const std::string NET_DIR     = "network_folder/";
const std::string PRIVATE_DIR = "client_private/";
const uint32_t NUM_WEIGHTS    = 55; // Matches the server's tracking constraints
int main() {
    // 1. Load CryptoContext
    CryptoContext<DCRTPoly> cc;
    if (!Serial::DeserializeFromFile(NET_DIR + "cc.bin", cc, SerType::BINARY)) {
        std::cerr << "ERROR: failed to deserialize cc.bin" << std::endl;
        return 1;
    }
    // 2. Load Private Key
    PrivateKey<DCRTPoly> secretKey;
    if (!Serial::DeserializeFromFile(PRIVATE_DIR + "sec.bin", secretKey, SerType::BINARY)) {
        std::cerr << "ERROR: failed to deserialize sec.bin" << std::endl;
        return 1;
    }
    // 3. Load both artifact outputs from the Server
    Ciphertext<DCRTPoly> encryptedSums;
    Ciphertext<DCRTPoly> encryptedCounts;
    if (!Serial::DeserializeFromFile(NET_DIR + "global_robust_aggregate.bin", encryptedSums, SerType::BINARY)) {
        std::cerr << "ERROR: failed to deserialize global_robust_aggregate.bin" << std::endl;
        return 1;
    }
    if (!Serial::DeserializeFromFile(NET_DIR + "global_contribution_counts.bin", encryptedCounts, SerType::BINARY)) {
        std::cerr << "ERROR: failed to deserialize global_contribution_counts.bin" << std::endl;
        return 1;
    }
    // 4. Decrypt both payloads
    Plaintext ptxtSums;
    Plaintext ptxtCounts;

    cc->Decrypt(secretKey, encryptedSums, &ptxtSums);
    cc->Decrypt(secretKey, encryptedCounts, &ptxtCounts);
    // Set lengths to look only at the 55 active payload slots
    ptxtSums->SetLength(NUM_WEIGHTS);
    ptxtCounts->SetLength(NUM_WEIGHTS);
    std::vector<double> rawSums    = ptxtSums->GetRealPackedValue();
    std::vector<double> rawCounts  = ptxtCounts->GetRealPackedValue();

    // 5. Compute Final Robust Average Weights (Sum / Count)
    std::cout << "\nCLIENT RECEIVED FINAL ROBUST AGGREGATED WEIGHTS:" << std::endl;
    std::cout << "------------------------------------------------" << std::endl;

    for (uint32_t i = 0; i < NUM_WEIGHTS; ++i) {
        // Round counts to nearest integer since CKKS introduces slight floating-point noise
        int count = static_cast<int>(std::round(rawCounts[i]));

        std::cout << "Slot [" << i << "]: ";
        if (count > 0) {
            double robustAverage = rawSums[i] / rawCounts[i];
            std::cout << robustAverage << "  (Compiled from " << count << " valid hospitals)" << std::endl;
        } else {
            std::cout << "0.0  (No hospitals contributed; all values flagged as outliers)" << std::endl;
        }
    }
    return 0;
}
