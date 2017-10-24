//
//  Variable.swift
//  SimpleNN
//
//  Created by taisuke fujita on 2017/08/17.
//  Copyright © 2017年 taisuke fujita. All rights reserved.
//

import Foundation

public protocol Variable: Codable {
    
    func filled(value: Double) -> Self
    
    func transpose() -> Self
    
    func scale(_ scalar: Double) -> Self
    
    func add(_ x: Double) -> Self
    
    func add(_ x: Self) -> Self
    
    func difference(_ x: Double) -> Self
    
    func difference(_ x: Self) -> Self
    
    // Hadamard product
    func multiply(_ x: Self) -> Self
}
