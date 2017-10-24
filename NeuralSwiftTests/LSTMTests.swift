//
//  LSTMTests.swift
//  NeuralSwiftTests
//
//  Created by taisuke fujita on 2017/10/19.
//  Copyright © 2017年 taisuke fujita. All rights reserved.
//

import XCTest
@testable import NeuralSwift

class LSTMTests: XCTestCase {
    
    override func setUp() {
        super.setUp()
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }
    
    override func tearDown() {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
        super.tearDown()
    }
    
    func testLSTM() {
        
        let wf = Matrix(array: [[0.7, 0.45]])
        let wi = Matrix(array: [[0.95, 0.8]])
        let wo = Matrix(array: [[0.6, 0.4]])
        let wa = Matrix(array: [[0.45, 0.25]])
        
        let rf = Matrix(array: [0.1])
        let ri = Matrix(array: [0.8])
        let ro = Matrix(array: [0.25])
        let ra = Matrix(array: [0.15])
        
        var lstmList = [LSTM]()
        for _ in 0..<2 {
            lstmList.append(LSTM(wf: wf, wi: wi, wo: wo, wa: wa, rf: rf, ri: ri, ro: ro, ra: ra))
        }
        
        let h = Vector(value: 0, rows: wf.rows)
        let cell = Vector(value: 0, rows: wf.rows)
        var state = (h, cell)
        let input = Vector(array: [1.0, 2.0])
        let input1 = Vector(array: [0.5, 3.0])
        
        // forward
        state = lstmList[0].forward(x: input, state: state)
        let expected0 = state.0.toArray()
        XCTAssertEqual(expected0[0], 0.53631, accuracy: 0.0001)
        let expected1 = state.1.toArray()
        XCTAssertEqual(expected1[0], 0.78572, accuracy: 0.0001)
        
        state = lstmList[1].forward(x: input1, state: state)
        let expected2 = state.0.toArray()
        XCTAssertEqual(expected2[0], 0.77198, accuracy: 0.0001)
        let expected3 = state.1.toArray()
        XCTAssertEqual(expected3[0], 1.5176, accuracy: 0.0001)
        
        // backward labels are [1, 2] -> 0.5, [0.5, 3]-> 1.25
        let label = Vector(array: [0.5])
        let label1 = Vector(array: [1.25])
        var deltaX = state.0 - label1
        
        let dY = Vector(value: 0, rows: wf.rows)
        let dCell = Vector(value: 0, rows: wf.rows)
        var bState = (dY, dCell)
        var next = LSTM(wf: wf, wi: wi, wo: wo, wa: wa, rf: rf, ri: ri, ro: ro, ra: ra)
        let backward = lstmList[1].backward(deltaX: deltaX, recurrentOut: bState, nextForget: next.forgetGateValue)
        let bExpected0 = backward.0.toArray()
        XCTAssertEqual(bExpected0[0], -0.01827, accuracy: 0.0001)
        
        deltaX = Vector(array: expected0) - label
        bState.0 = backward.0
        bState.1 = backward.1
        next = lstmList[1]
        let backward1 = lstmList[0].backward(deltaX: deltaX, recurrentOut: bState, nextForget: next.forgetGateValue)
        let bExpected1 = backward1.0.toArray()
        XCTAssertEqual(bExpected1[0], -0.00343, accuracy: 0.0001)
    }
    
    func _testLstmTrainSinWave() {
        /*
        let steps = 50
        let cycles = 100
        let length = steps * cycles + 1
        var samples = [[Double]]()
        var str = ""
        for n in 0..<length {
            // sin curve on 0 ot 1
            let data = [sin(Double(n) * (2 * Double.pi / Double(steps)))]
            samples.append(data)
            str += "\(data[0])\n"
        }
        */
        
        let length = 500
        var samples = [[Double]]()
        var str = ""
        for n in 0..<length {
            // sin curve on 0 ot 1
            let data = [sin(Double.pi / 180 * Double(n)) * 0.5 + 0.5]
            samples.append(data)
            str += "\(data)\n"
        }
        
        //print(str)
        
        let sequenceSize = 8
        let dataSize = 1
        
        let dataList = Array(samples.dropLast())
        let targetDataList = Array(samples.dropFirst())
        
        var rnn = LSTMNetwork(sequenceSize: sequenceSize,
                              inputDataLength: dataSize,
                              outputDataLength: dataSize,
                              learningRate: 0.1)
        let epoch = 1000
        var lossData = ""
        for e in 0..<epoch {
            let loss = rnn.train2(inputLists: dataList, targetLists: targetDataList)
            lossData += "\(loss)\n"
            print("Epoch:\(e)")
        }
        
        print(lossData)
        print("%%%%%%%%%%%%%%%%%")
        
        var str2 = ""
        let testList = dataList.chunks(sequenceSize)
        for test in testList {
            let results = rnn.query(lists: test)
            str2 += "\(results[results.count - 1][0])\n"
        }
        print(str2)
        
        /*
        var str2 = ""
        var test = [[0.5], [0.5087262032186417], [0.51744974835125046], [0.52616797812147187]]
        for _ in 0..<500 {
            test = rnn.query(lists: test)
            for result in test {
                str2 += "\(result[0])\n"
            }
        }
        print(str2)*/
    }
    
    func _testLstmTrain() {
        
        let steps = 50
        let cycles = 100
        let length = steps * cycles + 1
        var samples = [[Double]]()
        var str = ""
        for n in 0..<length {
            // sin curve on 0 ot 1
            let data = [sin(Double(n) * (2 * Double.pi / Double(steps)))]
            samples.append(data)
            str += "\(data)\n"
        }
        
        //print(str)
        
        let sequenceSize = 8
        let dataSize = 1
        
        let dataList = Array(samples.dropLast()).chunks(sequenceSize)
        let targetDataList = Array(samples.dropFirst()).chunks(sequenceSize)
        
        var rnn = LSTMNetwork(sequenceSize: sequenceSize,
                              inputDataLength: dataSize,
                              outputDataLength: dataSize,
                              learningRate: 0.03)
        let epoch = 100
        var lossData = ""
        for e in 0..<epoch {
            
            for i in 0..<dataList.count {
                let data = dataList[i]
                let targetData = targetDataList[i]
                let loss = rnn.train(inputLists: data, targetLists: targetData)
                lossData += "\(loss)\n"
            }
            print("Epoch:\(e)")
        }
        
        //print(lossData)
        print("%%%%%%%%%%%%%%%%%")
        
        var str2 = ""
        for i in 0..<dataList.count {
            let test = dataList[i]
            let results = rnn.query(lists: test)
            str2 += "\(results[sequenceSize - 1][0])\n"
        }
        print(str2)
    }
    
    func _testLstmTrain2() {
        
        let steps = 50
        let cycles = 100
        let length = steps * cycles + 1
        var samples = [Double]()
        var str = ""
        for n in 0..<length {
            // sin curve on 0 ot 1
            let data = sin(Double(n) * (2 * Double.pi / Double(steps)))
            samples.append(data)
            str += "\(data)\n"
        }
        
        //print(str)
        
        let sequenceSize = 8
        let dataSize = 1
        
        let t0 = Array(samples.dropLast(sequenceSize - 1))
        let t1 = Array(samples.dropFirst())
        let t2 = Array(samples.dropFirst(2))
        let t3 = Array(samples.dropFirst(3))
        let t4 = Array(samples.dropFirst(4))
        let t5 = Array(samples.dropFirst(5))
        let t6 = Array(samples.dropFirst(6))
        let t7 = Array(samples.dropFirst(sequenceSize - 1))
        var data = [[[Double]]]()
        
        for i in 0..<t0.count {
            data.append([[t0[i]], [t1[i]], [t2[i]], [t3[i]], [t4[i]], [t5[i]], [t6[i]], [t7[i]]])
        }
        
        let dataList = Array(data.dropLast())
        let targetDataList = Array(data.dropFirst())
        
        var rnn = LSTMNetwork(sequenceSize: sequenceSize,
                              inputDataLength: dataSize,
                              outputDataLength: dataSize,
                              learningRate: 0.03)
        let epoch = 100
        var lossData = ""
        for e in 0..<epoch {
            for i in 0..<dataList.count {
                let data = dataList[i]
                let targetData = targetDataList[i]
                let loss = rnn.train(inputLists: data, targetLists: targetData)
                lossData += "\(loss)\n"
            }
            print("Epoch:\(e)")
        }
        
        //print(lossData)
        print("%%%%%%%%%%%%%%%%%")
        
        var str2 = ""
        for i in 0..<(length - sequenceSize) {
            let test = dataList[i]
            let results = rnn.query(lists: test)
            str2 += "\(results[sequenceSize - 1][0])\n"
        }
        print(str2)
    }
    
    func _testLstmTrainZundoko() {
        
        
        
        // ズン ズン ズン ズン ドコ ドコ ドコ ズン
        let x: [[Double]] = [[0,1], [0,1], [0,1], [0,1], [1,0], [1,0], [1,0], [0,1]]
        
        var lstm = LSTMNetwork(sequenceSize: x.count,
                               inputDataLength: 2,
                               outputDataLength: 1,
                               learningRate: 0.03)
        
        
        let result = lstm.query(lists: x)
        print(result)
        print(result.map { $0.index(of: $0.max()!)! })
        print("dd")
        
        var kiyoshi = Kiyoshi()
        for _ in 0..<1000 {
            guard let k = kiyoshi.next() else {
                continue
            }
            print(k)
        }
    }
    
}

struct Kiyoshi: IteratorProtocol {
    
    var element: ([Int], Int) = ([0, 0], 0)
    
    let answer = [0, 0, 0, 0, 1]
    let src = [0, 0, 0, 0, 1]
    
    var x = [0, 0, 0, 0, 1]
    var y = [0, 0, 0, 0, 1]
    
    mutating func next() -> ([Int], Int)? {
        defer {
            let randomX = src[Int(arc4random_uniform(UInt32(src.count)))]
            x.append(randomX)
            y.append(x == answer ? 1 : 0)
        }
        let newX = x.remove(at: 0) == 0 ? [0, 1] : [1, 0]
        let newY = y.remove(at: 0)
        return (newX, newY)
    }
}

