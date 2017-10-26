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
        
        let bf = Vector(value: 0.15, rows: rf.rows)
        let bi = Vector(value: 0.65, rows: ri.rows)
        let bo = Vector(value: 0.1, rows: ro.rows)
        let ba = Vector(value: 0.2, rows: ra.rows)
        
        var lstmList = [LSTM]()
        for _ in 0..<2 {
            lstmList.append(LSTM(wf: wf, wi: wi, wo: wo, wa: wa,
                                 rf: rf, ri: ri, ro: ro, ra: ra,
                                 bf: bf, bi: bi, bo: bo, ba: ba))
        }
        
        let h = Vector(value: 0, rows: wf.rows)
        let cell = Vector(value: 0, rows: wf.rows)
        let state = (h, cell)
        let input = Vector(array: [1.0, 2.0])
        let input1 = Vector(array: [0.5, 3.0])
        
        // Forward
        let forward = lstmList[0].forward(x: input, state: state)
        let expected0 = forward.0.toArray()
        XCTAssertEqual(expected0[0], 0.53631, accuracy: 0.0001)
        let expected1 = forward.1.toArray()
        XCTAssertEqual(expected1[0], 0.78572, accuracy: 0.0001)
        
        let forward1 = lstmList[1].forward(x: input1, state: forward)
        let expected2 = forward1.0.toArray()
        XCTAssertEqual(expected2[0], 0.77198, accuracy: 0.0001)
        let expected3 = forward1.1.toArray()
        XCTAssertEqual(expected3[0], 1.5176, accuracy: 0.0001)
        
        // Backward labels are [1, 2] -> 0.5, [0.5, 3]-> 1.25
        let label = Vector(array: [0.5])
        let label1 = Vector(array: [1.25])
        var deltaX = forward1.0 - label1
        
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
    
    func testLSTMNetwork() {
        let inputData = [[1.0, 2.0], [0.5, 3.0]]
        let lstmNetwork = LSTMNetwork(sequenceSize: inputData.count,
                              inputDataLength: 2,
                              outputDataLength: 1,
                              learningRate: 0.1)
        
        lstmNetwork.wf = Matrix(array: [[0.7, 0.45]])
        lstmNetwork.wi = Matrix(array: [[0.95, 0.8]])
        lstmNetwork.wo = Matrix(array: [[0.6, 0.4]])
        lstmNetwork.wa = Matrix(array: [[0.45, 0.25]])
        
        lstmNetwork.rf = Matrix(array: [0.1])
        lstmNetwork.ri = Matrix(array: [0.8])
        lstmNetwork.ro = Matrix(array: [0.25])
        lstmNetwork.ra = Matrix(array: [0.15])
        
        lstmNetwork.bf = Vector(value: 0.15, rows: lstmNetwork.rf.rows)
        lstmNetwork.bi = Vector(value: 0.65, rows: lstmNetwork.ri.rows)
        lstmNetwork.bo = Vector(value: 0.1, rows: lstmNetwork.ro.rows)
        lstmNetwork.ba = Vector(value: 0.2, rows: lstmNetwork.ra.rows)
        
        lstmNetwork.reset()
        
        let result = lstmNetwork.query(lists: inputData)
        
        XCTAssertEqual(result[0][0], 0.53631, accuracy: 0.0001)
        XCTAssertEqual(result[1][0], 0.77198, accuracy: 0.0001)

        let labels = [[0.5], [1.25]]
        _ = lstmNetwork.train(inputLists: inputData, targetLists: labels)
        
        let wa = lstmNetwork.wa.toArray()
        XCTAssertEqual(wa[0], 0.45267, accuracy: 0.0001)
        XCTAssertEqual(wa[1], 0.25922, accuracy: 0.0001)
        
        let wi = lstmNetwork.wi.toArray()
        XCTAssertEqual(wi[0], 0.95022, accuracy: 0.0001)
        XCTAssertEqual(wi[1], 0.80067, accuracy: 0.0001)
        
        let wf = lstmNetwork.wf.toArray()
        XCTAssertEqual(wf[0], 0.70031, accuracy: 0.0001)
        XCTAssertEqual(wf[1], 0.45189, accuracy: 0.0001)
        
        let wo = lstmNetwork.wo.toArray()
        XCTAssertEqual(wo[0], 0.60259, accuracy: 0.0001)
        XCTAssertEqual(wo[1], 0.41626, accuracy: 0.0001)
        
        let ra = lstmNetwork.ra.toArray()
        XCTAssertEqual(ra[0], 0.15104, accuracy: 0.0001)
        
        let ri = lstmNetwork.ri.toArray()
        XCTAssertEqual(ri[0], 0.80006, accuracy: 0.0001)
        
        let rf = lstmNetwork.rf.toArray()
        XCTAssertEqual(rf[0], 0.10034, accuracy: 0.0001)
        
        let ro = lstmNetwork.ro.toArray()
        XCTAssertEqual(ro[0], 0.25297, accuracy: 0.0001)
        
        let ba = lstmNetwork.ba.toArray()
        XCTAssertEqual(ba[0], 0.20364, accuracy: 0.0001)
        
        let bi = lstmNetwork.bi.toArray()
        XCTAssertEqual(bi[0], 0.65028, accuracy: 0.0001)
        
        let bf = lstmNetwork.bf.toArray()
        XCTAssertEqual(bf[0], 0.15063, accuracy: 0.0001)
        
        let bo = lstmNetwork.bo.toArray()
        XCTAssertEqual(bo[0], 0.10536, accuracy: 0.0001)
    }
    
    func testLstmTrainSinWave() {
        
        let steps = 500
        let cycles = 5
        let length = steps * cycles + 1
        var samples = [[Double]]()
        var str = ""
        for n in 0..<length {
            // sin curve on 0 ot 1
            let data = [sin(Double(n) * (2 * Double.pi / Double(steps))) * 0.5 + 0.5]
            samples.append(data)
            str += "\(data[0])\n"
        }
        //print(str)
        
        let sequenceSize = 8
        let dataSize = 1
        
        let dataList = Array(samples.dropLast())
        let targetDataList = Array(samples.dropFirst())
        
        let rnn = LSTMNetwork(sequenceSize: sequenceSize,
                              inputDataLength: dataSize,
                              outputDataLength: dataSize,
                              learningRate: 0.1)
        let epoch = 5//00000
        for e in 0..<epoch {
            let loss = rnn.train(inputLists: dataList, targetLists: targetDataList)
            if e % 100 == 0 {
                print("Epoch:\(e)")
            }
            print(loss)
        }
        
        print("%%%%%%%%%%%%%%%%%")
        
        /*
        var str2 = ""
        let testList = dataList.chunks(sequenceSize)
        for test in testList {
            let results = rnn.query(lists: test)
            str2 += "\(results[results.count - 1][0])\n"
        }
        print(str2)*/
        
        var tmp = [[[Double]]]()
        for i in 0..<sequenceSize {
            let a = Array(Array(dataList.dropFirst(i)).dropLast(sequenceSize - i))
            tmp.append(a)
        }
        var data = [[Double]]()
        for i in 0..<tmp[0].count {
            for j in 0..<sequenceSize {
                data.append(tmp[j][i])
            }
        }
        var str2 = ""
        let testList = data.chunks(sequenceSize)
        for test in testList {
            let results = rnn.query(lists: test)
            str2 += "\(results[results.count - 1][0])\n"
        }
        print(str2)
    }
    
    func testLstmTrainZundoco() {
        
        let sequenceSize = 8
        
        let lstm = LSTMNetwork(sequenceSize: sequenceSize,
                               inputDataLength: 2,
                               outputDataLength: 2,
                               learningRate: 0.1)
        
        var pattern = PatternGenerator()
        var data = [[Double]]()
        var target = [[Double]]()
        // create training data
        for _ in 0..<5000 {
            let k = pattern.next()!
            data.append(k.0)
            target.append(k.1)
        }
        let epoch = 100
        for e in 0..<epoch {
            let loss = lstm.train(inputLists: data, targetLists: target)
            if e % 100 == 0 {
                print("Epoch:\(e)")
            }
            print(loss)
        }
        
        print("-----1-----")
        // zun zun zun zun doco doco doco zun
        var x: [[Double]] = [[0,1], [0,1], [0,1], [0,1], [1,0], [1,0], [1,0], [0,1]]
        var y = lstm.query(lists: x)
        for i in 0..<x.count {
            print(x[i].index(of: x[i].max()!) == 0 ? "doco" : "zun")
            print(y[i].index(of: y[i].max()!) == 0 ? "" : "kiyoshi!")
        }
        var expected = [0, 0, 0, 0, 1, 0, 0, 0]
        for i in 0..<expected.count {
            XCTAssertEqual(expected[i], y[i].index(of: y[i].max()!)!)
        }
        print("-----2-----")
        x = [[0,1], [0,1], [0,1], [0,1], [0,1], [1,0], [0,1], [0,1]]
        y = lstm.query(lists: x)
        for i in 0..<x.count {
            print(x[i].index(of: x[i].max()!) == 0 ? "doco" : "zun")
            print(y[i].index(of: y[i].max()!) == 0 ? "" : "kiyoshi!")
        }
        expected = [0, 0, 0, 0, 0, 1, 0, 0]
        for i in 0..<expected.count {
            XCTAssertEqual(expected[i], y[i].index(of: y[i].max()!)!)
        }
        print("-----3-----")
        x = [[1,0], [1,0], [0,1], [0,1], [0,1], [0,1], [1,0], [0,1]]
        y = lstm.query(lists: x)
        for i in 0..<x.count {
            print(x[i].index(of: x[i].max()!) == 0 ? "doco" : "zun")
            print(y[i].index(of: y[i].max()!) == 0 ? "" : "kiyoshi!")
        }
        expected = [0, 0, 0, 0, 0, 0, 1, 0]
        for i in 0..<expected.count {
            XCTAssertEqual(expected[i], y[i].index(of: y[i].max()!)!)
        }
    }
}

struct PatternGenerator: IteratorProtocol {
    
    var element: ([Double], [Double]) = ([0, 0], [0, 0])
    
    let answer = [0, 0, 0, 0, 1]
    let src = [0, 0, 0, 0, 1]
    
    var x = [0, 0, 0, 0, 1]
    var y = [0, 0, 0, 0, 1]
    
    mutating func next() -> ([Double], [Double])? {
        defer {
            let randomX = src[Int(arc4random_uniform(UInt32(src.count)))]
            x.append(randomX)
            y.append(x == answer ? 1 : 0)
        }
        let newX = x.remove(at: 0) == 0 ? [0.0, 1.0] : [1.0, 0.0]
        let newY = y.remove(at: 0) == 0 ? [1.0, 0.0] : [0.0, 1.0]
        return (newX, newY)
    }
}

