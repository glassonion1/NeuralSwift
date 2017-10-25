//
//  Array+chunks.swift
//  NeuralSwift
//
//  Created by taisuke fujita on 2017/10/25.
//  Copyright © 2017年 taisuke fujita. All rights reserved.
//

import Foundation

extension Array {
    func chunks(_ chunkSize: Int) -> [[Element]] {
        return stride(from: 0, to: self.count, by: chunkSize).map {
            Array(self[$0..<Swift.min($0 + chunkSize, self.count)])
        }
    }
}
