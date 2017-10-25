//
//  SoftmaxLayer.swift
//  SimpleNN
//
//  Created by taisuke fujita on 2017/08/16.
//  Copyright © 2017年 taisuke fujita. All rights reserved.
//

import Foundation

public class SoftmaxLayer: Layer {
    
    public private(set) var value: Vector
    
    public init(value: Vector) {
        self.value = value
    }
    
    public func forward(x: Vector) {
        value = softmax(x)
    }
    
    // see: https://github.com/rasmusbergpalm/DeepLearnToolbox/issues/113
    // see: https://stats.stackexchange.com/questions/79454/softmax-layer-in-a-neural-network
    /*public func backward(y: Vector, delta: Vector) -> Vector {
        return delta / Double(y.rows)
    }*/
    
    public func backward(delta: Vector) {
        let deltaX = value.multiply(delta)
        let sumdx = sum(deltaX)
        value = deltaX - value * sumdx
    }
}
