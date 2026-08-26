#include "openfhe.h"
#include "binfhecontext.h"
#include "ciphertext-ser.h"
#include "cryptocontext-ser.h"
#include "key/key-ser.h"
#include "scheme/ckksrns/ckksrns-ser.h"
#include "binfhecontext-ser.h"
#include <fstream>
#include <iostream>
#include <cmath>
#include <vector>

using namespace lbcrypto;

const std::string NET_DIR            = "network_folder/";
const uint32_t NUM_HOSPITALS         = 3;
const uint32_t NUM_WEIGHTS           = 55; // 11 parameters * 5 diseases
const double OUTLIER_MULTIPLIER_K    = 1.5; // 'k' multiplier threshold

int main() {
    // =====================================================================
    // 1. Load CryptoContext, Public Keys, and Evaluation Keys
    // =====================================================================
    CryptoContext<DCRTPoly> cc;
    if (!Serial::DeserializeFromFile(NET_DIR + "cc.bin", cc, SerType::BINARY)) {
        std::cerr << "ERROR: failed to deserialize cc.bin" << std::endl;
        return 1;
    }

    PublicKey<DCRTPoly> publicKey;
    if (!Serial::DeserializeFromFile(NET_DIR + "pub.bin", publicKey, SerType::BINARY)) {
        std::cerr << "ERROR: failed to deserialize pub.bin" << std::endl;
        return 1;
    }

    std::ifstream multStream(NET_DIR + "mult.bin", std::ios::binary);
    if (!multStream.is_open() || !cc->DeserializeEvalMultKey(multStream, SerType::BINARY)) {
        std::cerr << "ERROR: failed to deserialize mult.bin" << std::endl;
        return 1;
    }

    std::ifstream autoStream(NET_DIR + "auto.bin", std::ios::binary);
    if (!autoStream.is_open() || !cc->DeserializeEvalAutomorphismKey(autoStream, SerType::BINARY)) {
        std::cerr << "ERROR: failed to deserialize auto.bin" << std::endl;
        return 1;
    }

    // =====================================================================
    // 2. Load FHEW Context & Scheme Switching Keys
    // =====================================================================
    auto ccLWE = std::make_shared<BinFHEContext>();
    if (!Serial::DeserializeFromFile(NET_DIR + "ccLWE.bin", *ccLWE, SerType::BINARY)) {
        std::cerr << "ERROR: failed to deserialize ccLWE.bin" << std::endl;
        return 1;
    }

    RingGSWACCKey refKey;
    LWESwitchingKey swKey;
    if (!Serial::DeserializeFromFile(NET_DIR + "fhew_ref.bin", refKey, SerType::BINARY) ||
        !Serial::DeserializeFromFile(NET_DIR + "fhew_sw.bin", swKey, SerType::BINARY)) {
        std::cerr << "ERROR: failed to deserialize default FHEW bootstrapping keys" << std::endl;
        return 1;
    }
    ccLWE->BTKeyLoad({refKey, swKey});

    uint32_t extraDim = 262144;
    RingGSWACCKey rKeyDim;
    LWESwitchingKey sKeyDim;
    if (Serial::DeserializeFromFile(NET_DIR + "fhew_ref_" + std::to_string(extraDim) + ".bin", rKeyDim, SerType::BINARY) &&
        Serial::DeserializeFromFile(NET_DIR + "fhew_sw_" + std::to_string(extraDim) + ".bin", sKeyDim, SerType::BINARY)) {
        ccLWE->BTKeyMapLoadSingleElement(extraDim, {rKeyDim, sKeyDim});
        std::cout << "SERVER: Loaded extra bootstrapping key for dim " << extraDim << std::endl;
    }

    cc->SetBinCCForSchemeSwitch(ccLWE);

    Ciphertext<DCRTPoly> swkFC;
    if (!Serial::DeserializeFromFile(NET_DIR + "swkfc.bin", swkFC, SerType::BINARY)) {
        std::cerr << "ERROR: failed to deserialize swkfc.bin" << std::endl;
        return 1;
    }
    cc->SetSwkFC(swkFC);

    PrivateKey<DCRTPoly> debugKey;
    bool debugAvailable = Serial::DeserializeFromFile("client_private/sec.bin", debugKey, SerType::BINARY);
    auto debugPrint = [&](std::string label, Ciphertext<DCRTPoly> ctxt) {
        if (!debugAvailable) return;
        Plaintext ptxt;
        cc->Decrypt(debugKey, ctxt, &ptxt);
        ptxt->SetLength(NUM_WEIGHTS);
        std::cout << "DEBUG [" << label << "]: " << ptxt->GetRealPackedValue() << std::endl;
    };

    std::cout << "SERVER: Keys and setups loaded successfully." << std::endl;

    // =====================================================================
    // 3. Load Encrypted Hospital Weights
    // =====================================================================
    std::vector<Ciphertext<DCRTPoly>> hospitalWeights(NUM_HOSPITALS);
    for (uint32_t i = 0; i < NUM_HOSPITALS; ++i) {
        std::string filename = NET_DIR + "filtered_final_weights" + std::to_string(i + 1) + ".bin";
        if (!Serial::DeserializeFromFile(filename, hospitalWeights[i], SerType::BINARY)) {
            std::cerr << "ERROR: Failed to load " << filename << std::endl;
            return 1;
        }
        std::cout << "-> Loaded encrypted weights for Hospital " << (i + 1) << std::endl;
    }

    const uint32_t slots = 256;
    std::vector<double> maskVec(slots, 0.0);
    for (uint32_t i = 0; i < NUM_WEIGHTS; ++i) maskVec[i] = 1.0;
    auto slotsMask = cc->MakeCKKSPackedPlaintext(maskVec);

    for (uint32_t i = 0; i < NUM_HOSPITALS; ++i) {
        hospitalWeights[i] = cc->EvalMult(hospitalWeights[i], slotsMask);
    }

    // =====================================================================
    // 4. Compute Inter-Hospital Slot-Wise Mean and Variance
    // =====================================================================
    std::cout << "SERVER: Computing parallel slot-wise Mean and Variance..." << std::endl;

    auto encryptedSum = hospitalWeights[0];
    for (uint32_t i = 1; i < NUM_HOSPITALS; ++i) {
        encryptedSum = cc->EvalAdd(encryptedSum, hospitalWeights[i]);
    }
    auto meanCtxt = cc->EvalMult(encryptedSum, 1.0 / static_cast<double>(NUM_HOSPITALS));
    debugPrint("Computed Slot-Wise Mean", meanCtxt);

    auto varianceCtxt = cc->EvalSub(hospitalWeights[0], meanCtxt);
    varianceCtxt = cc->EvalMult(varianceCtxt, varianceCtxt);
    for (uint32_t i = 1; i < NUM_HOSPITALS; ++i) {
        auto diff = cc->EvalSub(hospitalWeights[i], meanCtxt);
        auto diffSq = cc->EvalMult(diff, diff);
        varianceCtxt = cc->EvalAdd(varianceCtxt, diffSq);
    }
    varianceCtxt = cc->EvalMult(varianceCtxt, 1.0 / static_cast<double>(NUM_HOSPITALS));
    debugPrint("Computed Slot-Wise Variance", varianceCtxt);

    // =====================================================================
    // 5. Scheme Switching & Parallel Outlier Filtration
    // =====================================================================
    auto modulusLWE = ccLWE->GetParams()->GetLWEParams()->Getq().ConvertToInt();
    auto beta       = ccLWE->GetBeta().ConvertToInt();
    auto pLWE       = modulusLWE / (2 * beta);

    double scaleSign = static_cast<double>(pLWE) / 8.0;
    cc->EvalCompareSwitchPrecompute(pLWE, scaleSign, /*unit=*/false);

    auto scaledVariance = cc->EvalMult(varianceCtxt, OUTLIER_MULTIPLIER_K);

    std::vector<Ciphertext<DCRTPoly>> validFlags(NUM_HOSPITALS);
    std::vector<Ciphertext<DCRTPoly>> filteredWeights(NUM_HOSPITALS);

    std::cout << "SERVER: Running Scheme-Switching comparison for outliers..." << std::endl;
    for (uint32_t i = 0; i < NUM_HOSPITALS; ++i) {
        auto diff = cc->EvalSub(hospitalWeights[i], meanCtxt);
        auto diffSq = cc->EvalMult(diff, diff);

        auto outlierIndicator = cc->EvalCompareSchemeSwitching(diffSq, scaledVariance, slots, slots, pLWE, scaleSign, false);

        validFlags[i] = cc->EvalSub(1.0, outlierIndicator);
        filteredWeights[i] = cc->EvalMult(hospitalWeights[i], validFlags[i]);

        std::cout << "   -> Hospital " << (i + 1) << " outlier filtering complete." << std::endl;
        debugPrint("Hospital " + std::to_string(i+1) + " Clean Flags", validFlags[i]);
        debugPrint("Hospital " + std::to_string(i+1) + " Filtered Output", filteredWeights[i]);
    }

    // =====================================================================
    // 6. Global & Clustered Aggregation using G.txt and C.txt Masks
    // =====================================================================
    std::cout << "SERVER: Loading aggregation masks G.txt and C.txt from current directory..." << std::endl;

    std::vector<double> g_mask_raw(11, 0.0);
    std::vector<double> c_mask_raw(11, 0.0);

    // CRITICAL: Look in current working execution directory explicitly
    std::ifstream g_file("G.txt");
    std::ifstream c_file("C.txt");

    if (!g_file.is_open() || !c_file.is_open()) {
        std::cerr << "ERROR: Missing G.txt or C.txt in the current cpp directory." << std::endl;
        return 1;
    }

    for (int i = 0; i < 11; ++i) g_file >> g_mask_raw[i];
    for (int i = 0; i < 11; ++i) c_file >> c_mask_raw[i];
    g_file.close();
    c_file.close();

    // Stretch 11 parameters to 55 slots (5 disease values per parameter block)
    std::vector<double> g_mask_55(slots, 0.0);
    std::vector<double> c_mask_55(slots, 0.0);

    for (uint32_t param = 0; param < 11; ++param) {
        for (uint32_t disease = 0; disease < 5; ++disease) {
            uint32_t slot_idx = param * 5 + disease;
            g_mask_55[slot_idx] = g_mask_raw[param];
            c_mask_55[slot_idx] = c_mask_raw[param];
        }
    }

    auto gMaskPtxt = cc->MakeCKKSPackedPlaintext(g_mask_55);
    auto cMaskPtxt = cc->MakeCKKSPackedPlaintext(c_mask_55);

    // --- Compute Global Aggregation Component ---
    auto globalSum = filteredWeights[0];
    auto globalFlags = validFlags[0];
    for (uint32_t i = 1; i < NUM_HOSPITALS; ++i) {
        globalSum = cc->EvalAdd(globalSum, filteredWeights[i]);
        globalFlags = cc->EvalAdd(globalFlags, validFlags[i]);
    }
    auto globalAggregatedSum    = cc->EvalMult(globalSum, gMaskPtxt);
    auto globalAggregatedCounts = cc->EvalMult(globalFlags, gMaskPtxt);

    // --- Compute Clustered Aggregation Component ---
    auto clusterAggregatedSum    = cc->EvalMult(filteredWeights[0], cMaskPtxt);
    auto clusterAggregatedCounts = cc->EvalMult(validFlags[0], cMaskPtxt);

    for (uint32_t i = 1; i < NUM_HOSPITALS; ++i) {
        auto hospitalMaskedSum   = cc->EvalMult(filteredWeights[i], cMaskPtxt);
        auto hospitalMaskedFlags = cc->EvalMult(validFlags[i], cMaskPtxt);

        clusterAggregatedSum    = cc->EvalAdd(clusterAggregatedSum, hospitalMaskedSum);
        clusterAggregatedCounts = cc->EvalAdd(clusterAggregatedCounts, hospitalMaskedFlags);
    }

    // --- Combine Global and Clustered Paths into Final Output Vectors ---
    auto finalAggregatedSum    = cc->EvalAdd(globalAggregatedSum, clusterAggregatedSum);
    auto finalAggregatedCounts = cc->EvalAdd(globalAggregatedCounts, clusterAggregatedCounts);

    debugPrint("Final Aggregated Sum (Combined)", finalAggregatedSum);
    debugPrint("Final Contribution Counts (Combined)", finalAggregatedCounts);

    // =====================================================================
    // 7. Serialize Output Artifacts
    // =====================================================================
    if (!Serial::SerializeToFile(NET_DIR + "global_robust_aggregate.bin", finalAggregatedSum, SerType::BINARY) ||
        !Serial::SerializeToFile(NET_DIR + "global_contribution_counts.bin", finalAggregatedCounts, SerType::BINARY)) {
        std::cerr << "ERROR: Failed to serialize aggregate results to network folder." << std::endl;
        return 1;
    }

    std::cout << "SERVER SUCCESS: Dynamic G/C weight vectors and counts written successfully to network_folder/." << std::endl;
    return 0;
}
