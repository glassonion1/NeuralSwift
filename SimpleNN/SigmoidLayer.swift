//
//  SigmoidLayer.swift
//  SimpleNN
//
//  Created by taisuke fujita on 2017/08/16.
//  Copyright © 2017年 taisuke fujita. All rights reserved.
//

import Foundation

public struct SigmoidLayer: Layer {
    
    public init() {
    }
    
    // http://pythonskywalker.hatenablog.com/entry/2016/12/24/093705
    public func forward(x: Vector) -> Vector {
        return sigmoid(x)
    }
    
    public func backward(y: Vector, delta: Vector) -> Vector {
        return delta.multiply(y).multiply(1.0 - y)
    }
}
