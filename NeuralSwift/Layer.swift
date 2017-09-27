//
//  Layer.swift
//  SimpleNN
//
//  Created by taisuke fujita on 2017/08/16.
//  Copyright © 2017年 taisuke fujita. All rights reserved.
//

import Foundation

public protocol Layer {
    
    func forward(x: Vector) -> Vector
    
    func backward(y: Vector, delta: Vector) -> Vector
}
