#include "openfhe.h"
#include "ciphertext-ser.h"
#include "cryptocontext-ser.h"
#include "key/key-ser.h"
#include "scheme/ckksrns/ckksrns-ser.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <string>

using namespace lbcrypto;

const std::string NET_DIR = "network_folder/";

// Define the file names here. Change these when running for different hospitals.
const std::string INPUT_FILE  = "filtered_final_weights3.txt";
const std::string OUTPUT_BIN  = "filtered_final_weights3.bin";
const std::string META_BIN    = "filtered_final_weights3_meta.bin";

template <typename T>
bool SerializeOrDie(const std::string& path, const T& obj, const std::string& label) {
    if (!Serial::SerializeToFile(path, obj, SerType::BINARY)) {
        std::cerr << "ERROR: failed to serialize " << label << " to " << path << std::endl;
        return false;
    }
    std::cout << "   wrote " << path << "   (" << label << ")" << std::endl;
    return true;
}

// Helper function to split a string by a specific delimiter character
std::vector<std::string> split(const std::string& s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter)) {
        // Trim leading/trailing whitespace if any remains
        size_t first = token.find_first_not_of(" \t\n\r");
        if (first != std::string::npos) {
            size_t last = token.find_last_not_of(" \t\n\r");
            tokens.push_back(token.substr(first, (last - first + 1)));
        }
    }
    return tokens;
}

int main() {
    // 1. Read and parse the hospital weight text file
    std::ifstream infile(INPUT_FILE);
    if (!infile.is_open()) {
        std::cerr << "ERROR: Could not open input file: " << INPUT_FILE << std::endl;
        return 1;
    }

    std::string fileContents((std::istreambuf_iterator<char>(infile)), std::istreambuf_iterator<char>());
    infile.close();

    std::vector<double> weights;

    // Split by comma first (splits the parameters)
    std::vector<std::string> parameters = split(fileContents, ',');

    for (const auto& paramChunk : parameters) {
        // Split by space next (splits the 5 disease metrics within that parameter)
        std::istringstream iss(paramChunk);
        double val;
        while (iss >> val) {
            weights.push_back(val);
        }
    }

    uint32_t numRealValues = static_cast<uint32_t>(weights.size());
    std::cout << "Successfully parsed " << numRealValues << " weights from " << INPUT_FILE << std::endl;

    if (numRealValues != 55) {
        std::cout << "WARNING: Expected 55 values (11 parameters x 5 diseases), but found " << numRealValues << ". Proceeding anyway..." << std::endl;
    }

    // 2. Load context and public key
    CryptoContext<DCRTPoly> cc;
    if (!Serial::DeserializeFromFile(NET_DIR + "cc.bin", cc, SerType::BINARY)) {
        std::cerr << "ERROR: failed to load cc.bin" << std::endl;
        return 1;
    }

    PublicKey<DCRTPoly> publicKey;
    if (!Serial::DeserializeFromFile(NET_DIR + "pub.bin", publicKey, SerType::BINARY)) {
        std::cerr << "ERROR: failed to load pub.bin" << std::endl;
        return 1;
    }

    // 3. Match capacity to CKKS context slots size and pad with 0.0
    const uint32_t slots = 256;
    weights.resize(slots, 0.0);

    // 4. Encrypt the structured vector
    Plaintext ptxt = cc->MakeCKKSPackedPlaintext(weights);
    auto encryptedWeights = cc->Encrypt(publicKey, ptxt);

    // 5. Save the output artifacts
    if (!SerializeOrDie(NET_DIR + OUTPUT_BIN, encryptedWeights, "encrypted hospital weights")) {
        return 1;
    }

    // Save metadata tracking the cleartext vector payload boundaries
    {
        std::ofstream metaFile(NET_DIR + META_BIN, std::ios::out | std::ios::binary);
        if (metaFile.is_open()) {
            metaFile.write(reinterpret_cast<const char*>(&numRealValues), sizeof(numRealValues));
            metaFile.close();
            std::cout << "   wrote " << NET_DIR + META_BIN << "   (meta count)" << std::endl;
        } else {
            std::cerr << "ERROR: Failed to write metadata configuration file." << std::endl;
            return 1;
        }
    }

    std::cout << "CLIENT SUCCESS: Data for " << INPUT_FILE << " successfully encrypted.\n" << std::endl;
    return 0;
}

