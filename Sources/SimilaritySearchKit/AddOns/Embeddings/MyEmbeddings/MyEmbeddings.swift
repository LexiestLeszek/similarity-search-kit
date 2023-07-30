//
//  MyEmbeddings.swift
//  
//
//  Created by Leszek Mielnikow on 27/07/2023.
//

import Foundation
import CoreML
import SimilaritySearchKit

@available(macOS 13.0, iOS 16.0, *)
public class MyEmbeddings: EmbeddingsProtocol {
    public let model: msmarco_bert
// The model is sentence-transformers/msmarco-bert-base-dot-v5 that I put through Zach's converting script

    public let tokenizer: BertTokenizer
    public let inputDimention: Int = 512
    public let outputDimention: Int = 768

    public init() {
        let modelConfig = MLModelConfiguration()
        modelConfig.computeUnits = .all

        do {
            self.model = try msmarco_bert(configuration: modelConfig)
        } catch {
            fatalError("Failed to load the Core ML model. Error: \(error.localizedDescription)")
        }

        self.tokenizer = BertTokenizer()
    }

    // MARK: - Dense Embeddings

    public func encode(sentence: String) async -> [Float]? {
        // Encode input text as bert tokens
        let inputTokens = tokenizer.buildModelTokens(sentence: sentence)
        let (inputIds, attentionMask) = tokenizer.buildModelInputs(from: inputTokens)

        print("Sentence: \(sentence)")
        print("Input Tokens: \(inputTokens)")
        print("Input IDs: \(inputIds)")
        print("Attention Mask: \(attentionMask)")
        // Send tokens through the MLModel
        let embeddings = generateMyEmbeddings(inputIds: inputIds, attentionMask: attentionMask)

        return embeddings
    }

    public func generateMyEmbeddings(inputIds: MLMultiArray, attentionMask: MLMultiArray) -> [Float]? {
        let inputFeatures = msmarco_bertInput(
            input_ids: inputIds,
            attention_mask: attentionMask
        )

        let output = try? model.prediction(input: inputFeatures)

        guard let embeddings = output?.embeddings else {
            return nil
        }

        let embeddingsArray: [Float] = (0..<embeddings.count).map { Float(embeddings[$0].floatValue) }

        return embeddingsArray
    }
}

