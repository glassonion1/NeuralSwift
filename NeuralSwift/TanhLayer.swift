//
//  TanhLayer.swift
//  SimpleNN
//
//  Created by taisuke fujita on 2017/08/17.
//  Copyright © 2017年 taisuke fujita. All rights reserved.
//

import Foundation

public class TanhLayer: Layer {
    
    public private(set) var value: Vector
    
    public init(value: Vector) {
        self.value = value
    }
    
    public func forward(x: Vector) {
        value = tanh(x)
    }
    
    public func backward(delta: Vector) {
        value = delta.multiply(1.0 - square(value))
    }
}
