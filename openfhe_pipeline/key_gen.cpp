#include "openfhe.h"
#include "binfhecontext.h"
#include "ciphertext-ser.h"
#include "cryptocontext-ser.h"
#include "key/key-ser.h"
#include "scheme/ckksrns/ckksrns-ser.h"
#include "binfhecontext-ser.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

using namespace lbcrypto;

const std::string NET_DIR     = "network_folder/";
const std::string PRIVATE_DIR = "client_private/";

// Scaled up to 256 to safely hold your 165 elements layout
const uint32_t    SLOTS       = 256;

template <typename T>
bool SerializeOrDie(const std::string& path, const T& obj, const std::string& label) {
    if (!Serial::SerializeToFile(path, obj, SerType::BINARY)) {
        std::cerr << "ERROR: failed to serialize " << label << " to " << path << std::endl;
        return false;
    }
    std::cout << "   wrote " << path << "   (" << label << ")" << std::endl;
    return true;
}

int main() {
    std::filesystem::create_directories(NET_DIR);
    std::filesystem::create_directories(PRIVATE_DIR);

    bool ok = true;

    // =====================================================================
    // 1. Build the CKKS context
    // =====================================================================
    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(15);
    parameters.SetScalingModSize(50);
    parameters.SetFirstModSize(60);
    parameters.SetScalingTechnique(FLEXIBLEAUTO);
    parameters.SetSecurityLevel(HEStd_NotSet); // Explicit RingDim required when NotSet
    parameters.SetRingDim(32768);             // Safe manual size for depth 15
    parameters.SetBatchSize(SLOTS);
    parameters.SetKeySwitchTechnique(HYBRID);

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    cc->Enable(SCHEMESWITCH);

    // Generate Base Public/Private Key pair
    auto keys = cc->KeyGen();

    // Generate Multiplication and Summation evaluation keys
    cc->EvalMultKeyGen(keys.secretKey);
    cc->EvalSumKeyGen(keys.secretKey);

    // FIX: Generate the Automorphism keys so "auto.bin" serialization succeeds.
    // These power-of-2 index steps allow vector rotations for mean/variance calculations.
    std::vector<int32_t> rotationIndices = {1, 2, 4, 8, 16, 32, 64, 128, -1, -2, -4, -8, -16, -32, -64, -128};
    cc->EvalAtIndexKeyGen(keys.secretKey, rotationIndices);

    // =====================================================================
    // 2. Scheme-switching setup
    // =====================================================================
    SchSwchParams schParams;
    schParams.SetSecurityLevelCKKS(HEStd_NotSet);
    schParams.SetSecurityLevelFHEW(STD128); // Required active for noise tracking
    schParams.SetCtxtModSizeFHEWLargePrec(25);
    schParams.SetNumSlotsCKKS(SLOTS);
    schParams.SetNumValues(SLOTS);

    LWEPrivateKey privateKeyFHEW = cc->EvalSchemeSwitchingSetup(schParams);
    auto ccLWE                   = cc->GetBinCCForSchemeSwitch();

    cc->EvalSchemeSwitchingKeyGen(keys, privateKeyFHEW);
    ccLWE->BTKeyGen(privateKeyFHEW);

    std::cout << "CKKS ring dimension : " << cc->GetRingDimension() << std::endl;
    std::cout << "FHEW lattice param n: " << ccLWE->GetParams()->GetLWEParams()->Getn() << std::endl;

    // =====================================================================
    // 3. Serialize everything to disk
    // =====================================================================
    ok &= SerializeOrDie(NET_DIR + "cc.bin", cc, "CKKS crypto context");
    ok &= SerializeOrDie(NET_DIR + "pub.bin", keys.publicKey, "CKKS public key");

    {
        std::ofstream multFile(NET_DIR + "mult.bin", std::ios::out | std::ios::binary);
        if (!multFile.is_open() || !cc->SerializeEvalMultKey(multFile, SerType::BINARY)) {
            std::cerr << "ERROR: failed to serialize mult.bin" << std::endl;
            ok = false;
        }
    }
    {
        std::ofstream autoFile(NET_DIR + "auto.bin", std::ios::out | std::ios::binary);
        if (!autoFile.is_open() || !cc->SerializeEvalAutomorphismKey(autoFile, SerType::BINARY)) {
            std::cerr << "ERROR: failed to serialize auto.bin" << std::endl;
            ok = false;
        }
    }
    {
        std::ofstream sumFile(NET_DIR + "sum.bin", std::ios::out | std::ios::binary);
        if (!sumFile.is_open() || !cc->SerializeEvalSumKey(sumFile, SerType::BINARY)) {
            std::cerr << "ERROR: failed to serialize sum.bin" << std::endl;
            ok = false;
        }
    }

    ok &= SerializeOrDie(NET_DIR + "ccLWE.bin", *ccLWE, "FHEW crypto context");
    ok &= SerializeOrDie(NET_DIR + "fhew_ref.bin", ccLWE->GetRefreshKey(), "FHEW refresh key (default)");
    ok &= SerializeOrDie(NET_DIR + "fhew_sw.bin", ccLWE->GetSwitchKey(), "FHEW switch key (default)");

    {
        auto btKeyMap = ccLWE->GetBTKeyMap();
        std::ofstream dimsFile(NET_DIR + "btkey_dims.txt");
        for (const auto& pair : *btKeyMap) {
            uint32_t dim = pair.first;
            dimsFile << dim << "\n";
            ok &= SerializeOrDie(NET_DIR + "fhew_ref_" + std::to_string(dim) + ".bin",
                                  pair.second.BSkey, "FHEW refresh key dim " + std::to_string(dim));
            ok &= SerializeOrDie(NET_DIR + "fhew_sw_" + std::to_string(dim) + ".bin",
                                  pair.second.KSkey, "FHEW switch key dim " + std::to_string(dim));
        }
    }

    ok &= SerializeOrDie(NET_DIR + "swkfc.bin", cc->GetSwkFC(), "FHEW->CKKS switching key");
    ok &= SerializeOrDie(PRIVATE_DIR + "sec.bin", keys.secretKey, "CKKS secret key (client-only)");

    if (!ok) {
        std::cerr << "\nOne or more files failed to serialize." << std::endl;
        return 1;
    }
    std::cout << "\nCLIENT: Keys successfully generated and verified." << std::endl;
    return 0;
}
