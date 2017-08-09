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
        assert(m1[0, 0] == 1.0)
        assert(m1[0, 1] == 2.0)
        assert(m1[0, 2] == 3.0)
        assert(m1[1, 2] == 6.0)
        print("------------")
        
        let m2 = m1 * 2
        print(m2.description)
        assert(m2[0, 0] == 1.0 * 2)
        assert(m2[1, 2] == 6.0 * 2)
        print("------------")
        
        let m3 = m1 + 10
        print(m3.description)
        assert(m3[0, 0] == 11.0)
        assert(m3[1, 2] == 16.0)
        print("------------")
        
        let m4 = m1 - 10
        print(m4.description)
        assert(m4[0, 0] == -9.0)
        assert(m4[1, 2] == -4.0)
        print("------------")
        
        let v1 = Matrix(array: [1.0, 2.0, 3.0])
        print(v1.description)
        assert(v1[0] == 1.0)
        assert(v1[1] == 2.0)
        assert(v1[2] == 3.0)
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
        assert(v2[0, 0] == 26.0)
        assert(v2[0, 1] == 23.0)
        assert(v2[0, 2] == 47.0)
    }
    
    func testMatrix3() {
        let m1 = Matrix(array: [[1.0, 3.0, 7.0],
                                [9.0, 5.0, 2.0],
                                [8.0, 4.0, 6.0]])
        let v1 = Matrix(array: [2.0, 4.0, 3.0])
        print("-------")
        let v2 = m1.dot(v1)
        print(v2.description)
        assert(v2[0] == 35.0)
        assert(v2[1] == 44.0)
        assert(v2[2] == 50.0)
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
        assert(minValue >= -0.5)
        assert(maxValue <= 0.5)
        
        let m1 = Matrix(rows: 3, cols: 3)
        print(m1.description)
        m1.toArray().forEach { scalar in
            assert(scalar >= -0.5)
            assert(scalar <= 0.5)
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
    
    func testSoftmax() {
        let test = Vector(array: [0.1,-0.9,0.3, 0.7])
        
        let softmaxed = softmax(test)
        print("&&&&&testSoftmax&&&&&")
        print(softmaxed)
        
        let sum = softmaxed.toArray().reduce(0.0, +)
        XCTAssertEqualWithAccuracy(sum, 1.0, accuracy: 0.000001)
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
    
    func testTransmission() {
        let nn = NeuralNetwork(inputLayerSize: 3, hiddenLayerSize: 3, outputLayerSize: 3, learningRate: 0.3)
        let weight = Matrix(array: [[0.9, 0.3, 0.4], [0.2, 0.8, 0.2], [0.1, 0.5, 0.6]])
        let inputs = Vector(array: [0.9, 0.1, 0.8])
        let outputs = nn.forward(weight: weight, inputs: inputs)
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
        let nn = NeuralNetwork(inputLayerSize: 3, hiddenLayerSize: 3, outputLayerSize: 3, learningRate: 0.1)
        let inputs = Vector(array: [0.4, 0.4, 0.4])
        let outputs = sigmoid(Vector(array: [2.3, 2.3, 2.3]))
        let errors = Vector(array: [0.8, 0.8, 0.8])
        let results = nn.backward(inputs: inputs, outputs: outputs, errors: errors)
        print(results)
        XCTAssertEqualWithAccuracy(results[0, 0], 0.002650, accuracy: 0.000001)
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
            // training
            let trainingCsv = try String(contentsOfFile: trainingPath, encoding: String.Encoding.utf8)
            let trainingList = trainingCsv.components(separatedBy: .newlines).map { $0.components(separatedBy: ",") }.dropLast()
            assert(trainingList.count == 100)
            trainingList.forEach { data in
                assert(data.count == 28 * 28 + 1)
                let answer = Int(data[0])!
                let inputs = data.dropFirst().map { (Double($0)! / 255.0 * 0.99) + 0.01 }
                assert(inputs.count == 28 * 28)
                var targets = [Double](repeating: 0.01, count: 10)
                targets[answer] = 0.99
                nn.train(inputList: inputs, targetList: targets)
            }
            // test
            let testCsv = try String(contentsOfFile: testPath, encoding: String.Encoding.utf8)
            let testList = testCsv.components(separatedBy: .newlines).map { $0.components(separatedBy: ",") }.dropLast()
            assert(testList.count == 10)
            
            var scoreCard = [Int]()
            testList.forEach { test in
                assert(test.count == 28 * 28 + 1)
                let answer = Int(test[0])!
                let inputs = test.dropFirst().map { (Double($0)! / 255.0 * 0.99) + 0.01 }
                assert(inputs.count == 28 * 28)
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
            assert(sum >= 5)
        } catch {
            assertionFailure()
        }
    }
    
    func testRnnQuery() {
        let data = [[1.0, 2.0], [0.5, 3.0]]
        let target = [[0.5], [1.25]]
        var rnn = LSTMLayer(sequenceSize: data.count, inputDataLength: data[0].count, outputDataLength: target[0].count, learningRate: 0.1)
        rnn.wa = Matrix(array: [[0.45, 0.25]])
        rnn.wi = Matrix(array: [[0.95, 0.8]])
        rnn.wf = Matrix(array: [[0.7, 0.45]])
        rnn.wo = Matrix(array: [[0.6, 0.4]])
        
        rnn.ra = Matrix(array: [[0.15]])
        rnn.ri = Matrix(array: [[0.8]])
        rnn.rf = Matrix(array: [[0.1]])
        rnn.ro = Matrix(array: [[0.25]])
        
        rnn.reset()
        
        //let results = rnn.query(lists: data)
        //print("----testRnnQuery----")
        //print(results)
        print("^^^^^^^^")
        print(rnn.wa)

        rnn.train(inputLists: data, targetLists: target)
        
        print("^^^^^^^^")
        print(rnn.wa)
    }
    
    func testRnnTrain() {
        let data: [[Double]] = [[0,0,1,0], [1,0,0,0], [0,1,0,0], [0,1,0,0], [0,0,1,0], [0,1,0,0]]
        let targetData: [[Double]] = [[0,1,0,0], [1,0,0,0], [1,0,0,0], [0,1,0,0], [0,1,0,0], [0,1,0,0]]
        var rnn = LSTMLayer(sequenceSize: data.count, inputDataLength: data[0].count, outputDataLength: data[0].count, learningRate: 0.05)
        
        print("^^^^^^^^")
        print(rnn.wa)
        let epoch = 100
        for _ in 0..<epoch {
            rnn.train(inputLists: data, targetLists: targetData)
        }
        print("^^^^^^^^")
        print(rnn.wa)
        
        let test: [[Double]] = [[0,0,1,0], [1,0,0,0], [0,1,0,0], [0,1,0,0], [0,0,1,0], [0,1,0,0]]
        let results = rnn.query(lists: test)
        
        print("----testRnnTrain----")
        print(results.map { softmax(Vector(array: $0)) })
    }
    
    
}
