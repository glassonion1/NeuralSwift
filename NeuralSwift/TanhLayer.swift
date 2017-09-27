//
//  TanhLayer.swift
//  SimpleNN
//
//  Created by taisuke fujita on 2017/08/17.
//  Copyright © 2017年 taisuke fujita. All rights reserved.
//

import Foundation

public struct TanhLayer: Layer {
    
    public init() {
    }
    
    public func forward(x: Vector) -> Vector {
        return tanh(x)
    }
    
    public func backward(y: Vector, delta: Vector) -> Vector {
        return delta.multiply(1.0 - square(y))
    }
}
