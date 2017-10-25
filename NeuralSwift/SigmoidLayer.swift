//
//  SigmoidLayer.swift
//  SimpleNN
//
//  Created by taisuke fujita on 2017/08/16.
//  Copyright © 2017年 taisuke fujita. All rights reserved.
//

import Foundation

public class SigmoidLayer: Layer {
    
    public private(set) var value: Vector
    
    public init(value: Vector) {
        self.value = value
    }
    
    // http://pythonskywalker.hatenablog.com/entry/2016/12/24/093705
    public func forward(x: Vector) {
        value = sigmoid(x)
    }
    
    public func backward(delta: Vector) {
        value = delta.multiply(value).multiply(1.0 - value)
    }
}
