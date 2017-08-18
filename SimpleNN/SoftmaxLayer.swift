//
//  SoftmaxLayer.swift
//  SimpleNN
//
//  Created by taisuke fujita on 2017/08/16.
//  Copyright © 2017年 taisuke fujita. All rights reserved.
//

import Foundation

public struct SoftmaxLayer: Layer {
    
    public init() {
    }
    
    public func forward(x: Vector) -> Vector {
        return softmax(x)
    }
    
    // see: https://github.com/rasmusbergpalm/DeepLearnToolbox/issues/113
    // see: https://stats.stackexchange.com/questions/79454/softmax-layer-in-a-neural-network
    /*public func backward(y: Vector, delta: Vector) -> Vector {
        return delta / Double(y.rows)
    }*/
    
    public func backward(y: Vector, delta: Vector) -> Vector {
        let deltaX = y.multiply(delta)
        let sumdx = sum(deltaX)
        return deltaX - y * sumdx
    }
}
