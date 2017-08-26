//
//  SimpleNNTests.swift
//  SimpleNNTests
//
//  Created by taisuke fujita on 2017/07/19.
//  Copyright © 2017年 taisuke fujita. All rights reserved.
//

import XCTest
import Accelerate
import GameplayKit
@testable import SimpleNN


extension Array {
    func chunks(_ chunkSize: Int) -> [[Element]] {
        return stride(from: 0, to: self.count, by: chunkSize).map {
            Array(self[$0..<Swift.min($0 + chunkSize, self.count)])
        }
    }
}

class SimpleNNTests: XCTestCase {
    
    override func setUp() {
        super.setUp()
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }
    
    override func tearDown() {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
        super.tearDown()
    }
    
    func testMatrix() {
        let m1 = Matrix(array: [[1.0, 2.0, 3.0],
                                [4.0, 5.0, 6.0],
                                [7.0, 8.0, 9.0]])
        print(m1.description)
        XCTAssert(m1[0, 0] == 1.0)
        XCTAssert(m1[0, 1] == 2.0)
        XCTAssert(m1[0, 2] == 3.0)
        XCTAssert(m1[1, 2] == 6.0)
        print("------------")
        
        let m2 = m1 * 2
        print(m2.description)
        XCTAssert(m2[0, 0] == 1.0 * 2)
        XCTAssert(m2[1, 2] == 6.0 * 2)
        print("------------")
        
        let m3 = m1 + 10
        print(m3.description)
        XCTAssert(m3[0, 0] == 11.0)
        XCTAssert(m3[1, 2] == 16.0)
        print("------------")
        
        let m4 = m1 - 10
        print(m4.description)
        XCTAssert(m4[0, 0] == -9.0)
        XCTAssert(m4[1, 2] == -4.0)
        print("------------")
        
        let v1 = Matrix(array: [1.0, 2.0, 3.0])
        print(v1.description)
        XCTAssert(v1[0] == 1.0)
        XCTAssert(v1[1] == 2.0)
        XCTAssert(v1[2] == 3.0)
        print("------------")
    }
    
    func testMatrix2() {
        let m1 = Matrix(array: [[2.0, 4.0, 6.0],
                                [3.0, 1.0, 5.0],
                                [8.0, 7.0, 9.0]])
        let v1 = Matrix(array: [3.0, 4.0, 1.0])
        print("-------")
        let v2 = v1.dot(m1)
        print(v2.description)
        XCTAssert(v2[0, 0] == 26.0)
        XCTAssert(v2[0, 1] == 23.0)
        XCTAssert(v2[0, 2] == 47.0)
    }
    
    func testMatrix3() {
        let m1 = Matrix(array: [[1.0, 3.0, 7.0],
                                [9.0, 5.0, 2.0],
                                [8.0, 4.0, 6.0]])
        let v1 = Matrix(array: [2.0, 4.0, 3.0])
        print("-------")
        let v2 = m1.dot(v1)
        print(v2.description)
        XCTAssert(v2[0] == 35.0)
        XCTAssert(v2[1] == 44.0)
        XCTAssert(v2[2] == 50.0)
    }
    
    func testRandomMatrix() {
        let count = 1000
        var sum = 0.0
        var squareSum = 0.0
        var minValue = 100.0
        var maxValue = -100.0
        let rand = GKGaussianDistribution(randomSource: GKMersenneTwisterRandomSource(),
                                          lowestValue: -500, highestValue: 500)
        for _ in 0..<count {
            let randomValue = Double(rand.nextInt()) / 1000.0
            sum += randomValue
            squareSum += randomValue * randomValue
            if randomValue < minValue {
                minValue = randomValue
            }
            if maxValue < randomValue {
                maxValue = randomValue
            }
        }
        let mean = sum / Double(count)
        let variance = squareSum / Double(count) - mean * mean
        
        print("----------")
        print("min: \(minValue)")
        print("max: \(maxValue)")
        print("mean: \(mean)")
        print("variance: \(variance)")
        XCTAssert(minValue >= -0.5)
        XCTAssert(maxValue <= 0.5)
        
        let m1 = Matrix(rows: 3, cols: 3)
        print(m1.description)
        m1.toArray().forEach { scalar in
            XCTAssert(scalar >= -0.5)
            XCTAssert(scalar <= 0.5)
        }
    }
    
    func testVector() {
        let a = Vector(array: [20, 30, 40])
        let b = Vector(array: [3, 1]).transpose()
        let c: Matrix = a * b
        print("¥¥¥¥¥¥¥¥")
        print(c)
        XCTAssertEqualWithAccuracy(c[0, 0], 60, accuracy: 0.001)
        XCTAssertEqualWithAccuracy(c[0, 1], 20, accuracy: 0.001)
        XCTAssertEqualWithAccuracy(c[1, 0], 90, accuracy: 0.001)
        XCTAssertEqualWithAccuracy(c[1, 1], 30, accuracy: 0.001)
        XCTAssertEqualWithAccuracy(c[2, 0], 120, accuracy: 0.001)
        XCTAssertEqualWithAccuracy(c[2, 1], 40, accuracy: 0.001)
    }
    
    func testVectorAdd() {
        let a = Vector(array: [20, 30])
        let b = Vector(array: [3, 1])
        let c: Vector = a + b
        print("add¥¥¥¥¥¥¥¥")
        print(c)
    }
    
    func testVectorSum() {
        let a = Vector(array: [20, 30.3, -9.0, -0.09])
        let result = sum(a)
        XCTAssertEqualWithAccuracy(result, 41.21, accuracy: 0.01)
    }
    
    func testSoftmax() {
        let test = Vector(array: [0.1,-0.9,0.3, 0.7])
        
        let softmaxed = softmax(test)
        print("&&&&&testSoftmax&&&&&")
        print(softmaxed)
        
        let sum = softmaxed.toArray().reduce(0.0, +)
        XCTAssertEqualWithAccuracy(sum, 1.0, accuracy: 0.000001)
    }
    
    func testTanhLayer() {
        let data = Vector(array: [-1.0, 3.3, 5.0, 0.2])
        let layer = TanhLayer()
        let result = layer.forward(x: data)
        print(result)
        XCTAssertEqualWithAccuracy(result[0], -0.76159415595576485, accuracy: 0.000000001)
        XCTAssertEqualWithAccuracy(result[1], 0.99728296009914208, accuracy: 0.000000001)
        XCTAssertEqualWithAccuracy(result[2], 0.99990920426259511, accuracy: 0.000000001)
        XCTAssertEqualWithAccuracy(result[3], 0.19737532022490401, accuracy: 0.000000001)
        
        let target = Vector(array: [-1.0, 3.3, 0.5, 0.2])
        let result2 = layer.backward(y: result, delta: result - target)
        print(result2)
        print("dddddd")
    }
    
    func testSigmoidLayer() {
        let data = Vector(array: [-1.0, 3.3, 5.0, 0.2])
        let layer = SigmoidLayer()
        let result = layer.forward(x: data)
        XCTAssertEqualWithAccuracy(result[0], 0.268941421369995, accuracy: 0.000000001)
        XCTAssertEqualWithAccuracy(result[1], 0.96442881, accuracy: 0.000000001)
        XCTAssertEqualWithAccuracy(result[2], 0.99330715, accuracy: 0.000000001)
        XCTAssertEqualWithAccuracy(result[3], 0.549833997312478, accuracy: 0.000000001)
        
        let target = Vector(array: [-1.0, 3.3, 0.5, 0.2])
        let result2 = layer.backward(y: result, delta: result - target)
        print(result2)
        print("dddddd")
    }
    
    func testSoftmaxLayer() {
        let data = Vector(array: [-1.0, 3.3, 5.0, 0.2])
        let layer = SoftmaxLayer()
        let result = layer.forward(x: data)
        print(result)
        XCTAssertEqualWithAccuracy(result[0], 0.0020770644753070051, accuracy: 0.000000001)
        XCTAssertEqualWithAccuracy(result[1], 0.15307922333088542, accuracy: 0.000000001)
        XCTAssertEqualWithAccuracy(result[2], 0.83794761527972594, accuracy: 0.000000001)
        XCTAssertEqualWithAccuracy(result[3], 0.0068960969140816957, accuracy: 0.000000001)
        let sum = result.toArray().reduce(0.0, +)
        XCTAssertEqualWithAccuracy(sum, 1.0, accuracy: 0.000001)
        
        let target = Vector(array: [-1.0, 3.3, 0.5, 0.2])
        print(result - target)
        let result2 = layer.backward(y: result, delta: result - target)
        print(result2)
        print("dddddd")
    }
    
    func testNNQuery() {
        let nn = NeuralNetwork(inputLayerSize: 3, hiddenLayerSize: 3, outputLayerSize: 3, learningRate: 0.3)
        let result = nn.query(list: [1.0, 0.5, -1.5])
        print("+++++++++++")
        print(result)
        print("+++++++++++")
        let nn2 = NeuralNetwork(inputLayerSize: 3, hiddenLayerSize: 3, outputLayerSize: 3, learningRate: 0.3)
        let result2 = nn2.query(list: [1.0, 0.5, -1.5])
        print("+++++++++++")
        print(result2)
        print("+++++++++++")
        XCTAssertNotEqualWithAccuracy(result[0], result2[0], 0.0001)
        XCTAssertNotEqualWithAccuracy(result[1], result2[1], 0.0001)
        XCTAssertNotEqualWithAccuracy(result[2], result2[2], 0.0001)
    }
    
    func testForward() {
        let weight = Matrix(array: [[0.9, 0.3, 0.4], [0.2, 0.8, 0.2], [0.1, 0.5, 0.6]])
        let inputs = Vector(array: [0.9, 0.1, 0.8])
        let layer = SigmoidLayer()
        let outputs = layer.forward(x: weight * inputs)
        XCTAssertEqualWithAccuracy(outputs[0], 0.761, accuracy: 0.001)
        XCTAssertEqualWithAccuracy(outputs[1], 0.603, accuracy: 0.001)
        XCTAssertEqualWithAccuracy(outputs[2], 0.650, accuracy: 0.001)
    }
    
    func testQuery() {
        var nn = NeuralNetwork(inputLayerSize: 3, hiddenLayerSize: 3, outputLayerSize: 3, learningRate: 0.3)
        nn.w1 = Matrix(array: [[0.9, 0.3, 0.4], [0.2, 0.8, 0.2], [0.1, 0.5, 0.6]])
        nn.w2 = Matrix(array: [[0.3, 0.7, 0.5], [0.6, 0.5, 0.2], [0.8, 0.1, 0.9]])
        let outputs = nn.query(list: [0.9, 0.1, 0.8])
        print(outputs)
        XCTAssertEqualWithAccuracy(outputs[0], 0.726, accuracy: 0.001)
        XCTAssertEqualWithAccuracy(outputs[1], 0.708, accuracy: 0.001)
        XCTAssertEqualWithAccuracy(outputs[2], 0.778, accuracy: 0.001)
    }
    
    func testBackpropagation() {
        let inputs = Vector(array: [0.4, 0.4, 0.4])
        let outputs = sigmoid(Vector(array: [2.3, 2.3, 2.3]))
        let errors = Vector(array: [0.8, 0.8, 0.8])
        let layer = SigmoidLayer()
        let results = layer.backward(y: outputs, delta: errors) * inputs.transpose()
        print(results)
        XCTAssertEqualWithAccuracy(results[0, 0], 0.02650, accuracy: 0.00001)
    }
    
    func testTrain() {
        let inputs = [0.4, 0.4, 0.4]
        let targets = [0.94, 0.45, 1.2]
        
        var nn = NeuralNetwork(inputLayerSize: inputs.count,
                               hiddenLayerSize: 3,
                               outputLayerSize: targets.count,
                               learningRate: 0.1)
        //nn.w1 = Matrix(array: [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0]])
        //nn.w2 = Matrix(array: [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0]])
        nn.train(inputList: inputs, targetList: targets)
        print("**********")
        print(nn.w1)
        print(nn.w2)
    }
    
    func testMnist() {
        guard let trainingPath = Bundle(for: SimpleNNTests.self).path(forResource: "mnist_train_100", ofType: "csv") else {
            assertionFailure()
            return
        }
        guard let testPath = Bundle(for: SimpleNNTests.self).path(forResource: "mnist_test_10", ofType: "csv") else {
            assertionFailure()
            return
        }
        var nn = NeuralNetwork(inputLayerSize: 28 * 28,
                               hiddenLayerSize: 100,
                               outputLayerSize: 10,
                               learningRate: 0.3)
        do {
            var lossData = ""
            // training
            let trainingCsv = try String(contentsOfFile: trainingPath, encoding: String.Encoding.utf8)
            let trainingList = trainingCsv.components(separatedBy: .newlines).map { $0.components(separatedBy: ",") }.dropLast()
            XCTAssert(trainingList.count == 100)
            
            let epoch = 1
            for _ in 0..<epoch {
                trainingList.forEach { data in
                    XCTAssert(data.count == 28 * 28 + 1)
                    let answer = Int(data[0])!
                    let inputs = data.dropFirst().map { (Double($0)! / 255.0 * 0.99) + 0.01 }
                    XCTAssert(inputs.count == 28 * 28)
                    var targets = [Double](repeating: 0.01, count: 10)
                    targets[answer] = 0.99
                    let loss = nn.train(inputList: inputs, targetList: targets)
                    lossData += "\(loss)\n"
                }
            }
            //print(lossData)
            
            // test
            let testCsv = try String(contentsOfFile: testPath, encoding: String.Encoding.utf8)
            let testList = testCsv.components(separatedBy: .newlines).map { $0.components(separatedBy: ",") }.dropLast()
            XCTAssert(testList.count == 10)
            
            var scoreCard = [Int]()
            testList.forEach { test in
                XCTAssert(test.count == 28 * 28 + 1)
                let answer = Int(test[0])!
                let inputs = test.dropFirst().map { (Double($0)! / 255.0 * 0.99) + 0.01 }
                XCTAssert(inputs.count == 28 * 28)
                let outputs = nn.query(list: inputs)
                let max = outputs.max()!
                let expect = outputs.index(of: max)!
                print("        Certainty: \(max)")
                print("Networks's answer: \(expect)")
                print("    Correct label: \(answer)")
                print("----")
                //assert(answer == expect)
                if answer == expect {
                    scoreCard.append(1)
                } else {
                    scoreCard.append(0)
                }
            }
            let sum = scoreCard.reduce(0, +)
            print(Float(sum) / Float(scoreCard.count))
            XCTAssert(sum >= 5)
        } catch {
            XCTAssert(false)
        }
    }
    
    func testLstmTrain() {
        
        // [1,1,2,3,1] [2,1,2,3,2] [3,1,2,3,3] [4,1,2,3,4] [5,1,2,3,5]
        let maxValue = 5
        
        let dataList: [[[Double]]] = [[0,1,2,3], [1,1,2,3], [2,1,2,3], [3,1,2,3], [4,1,2,3], [5,1,2,3]].map { toOneHot(values: $0, maxValue: maxValue) }
        let targetDataList: [[[Double]]] = [[1,2,3,0], [1,2,3,1], [1,2,3,2], [1,2,3,3], [1,2,3,4], [1,2,3,5]].map { toOneHot(values: $0, maxValue: maxValue) }
        var rnn = LSTMNetwork(sequenceSize: 4, inputDataLength: maxValue + 1, outputDataLength: maxValue + 1, learningRate: 0.1)
        
        
        let epoch = 1000
        var lossData = ""
        for _ in 0..<epoch {
            for i in 0..<dataList.count {
                let data = dataList[i]
                let targetData = targetDataList[i]
                let loss = rnn.train(inputLists: data, targetLists: targetData)
                lossData += "\(loss)\n"
            }
        }
        
        print(lossData)
        
        let test: [[Double]] = toOneHot(values: [0,1,2,3], maxValue: maxValue)
        let results = rnn.query(lists: test)
        
        //let expected = results.map { softmax(Vector(array: $0)) }.map { $0.toArray().index(of: $0.toArray().max()!)! }
        let expected = results.map { Vector(array: $0) }.map { $0.toArray().index(of: $0.toArray().max()!)! }

        print("----testRnnTrain2----")
        print(results)
        print(expected)
        
        XCTAssertTrue([1,2,3,0] == expected)
    }
}
