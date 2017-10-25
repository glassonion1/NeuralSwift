//
//  Layer.swift
//  SimpleNN
//
//  Created by taisuke fujita on 2017/08/16.
//  Copyright © 2017年 taisuke fujita. All rights reserved.
//

import Foundation

public protocol Layer {
    
    var value: Vector { get }
    
    func forward(x: Vector)
    
    func backward(delta: Vector)
}
