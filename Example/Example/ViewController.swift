//
//  ViewController.swift
//  Sample
//
//  Created by taisuke fujita on 2017/07/21.
//  Copyright © 2017年 Taisuke Fujita. All rights reserved.
//

import UIKit
import SimpleNN

class ViewController: UIViewController {

    @IBOutlet weak var state: UILabel!
    @IBOutlet weak var progress: UIProgressView!
    @IBOutlet weak var button: UIButton!
    
    override func viewDidLoad() {
        super.viewDidLoad()
    }

    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
    }
    
    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }

    @IBAction private func startTraining(_ sender: AnyObject) {
        
        state.text = "training start"
        
        button.isEnabled = false
        
        DispatchQueue(label: "xxx").async {
            self.train()
        }
    }
    
    func train() {
        guard let trainingPath = Bundle.main.path(forResource: "mnist_train", ofType: "csv") else {
            return
        }
        guard let testPath = Bundle.main.path(forResource: "mnist_test", ofType: "csv") else {
            return
        }
        var nn = NeuralNetwork(inputLayerSize: 28 * 28,
                               hiddenLayerSize: 100,
                               outputLayerSize: 10,
                               learningRate: 0.3)
        do {
            // training
            let trainingCsv = try String(contentsOfFile: trainingPath, encoding: String.Encoding.utf8)
            let trainingList = trainingCsv.components(separatedBy: .newlines).dropLast()
            print("prepareing training data: \(trainingList.count)")
            let list = trainingList.map { text -> (Int, [Double]) in
                let data = text.components(separatedBy: ",")
                let answer = Int(data[0])!
                let inputs = data.dropFirst().map { (Double($0)! / 255.0 * 0.99) + 0.01 }
                return (answer, inputs)
            }
            
            DispatchQueue.main.async {
                self.state.text = "training..."
            }
            
            let epoch = 2
            var count = 0
            for _ in 0..<epoch {
                list.forEach { (answer, inputs) in
                    var targets = [Double](repeating: 0.01, count: 10)
                    targets[answer] = 0.99
                    nn.train(inputList: inputs, targetList: targets)
                    count += 1
                    DispatchQueue.main.async {
                        self.progress.setProgress(Float(count) / Float(trainingList.count * epoch), animated: true)
                    }
                }
            }
            
            DispatchQueue.main.async {
                self.state.text = "training end"
            }
            
            // test
            let testCsv = try String(contentsOfFile: testPath, encoding: String.Encoding.utf8)
            let testList = testCsv.components(separatedBy: .newlines).map { $0.components(separatedBy: ",") }.dropLast()
            
            DispatchQueue.main.async {
                self.state.text = "Testing..."
                self.progress.setProgress(0.0, animated: true)
            }
            
            var testCount = 0
            var scoreCard = [Int]()
            testList.forEach { test in
                let answer = Int(test[0])!
                let inputs = test.dropFirst().map { (Double($0)! / 255.0 * 0.99) + 0.01 }
                let outputs = nn.query(list: inputs)
                let max = outputs.max()!
                let expect = outputs.index(of: max)!
                print("        Certainty: \(max)")
                print("Networks's answer: \(expect)")
                print("    Correct label: \(answer)")
                print("----")
                testCount += 1
                DispatchQueue.main.async {
                    self.progress.setProgress(Float(testCount) / Float(testList.count), animated: true)
                }
                if answer == expect {
                    scoreCard.append(1)
                } else {
                    scoreCard.append(0)
                }
            }
            let sum = scoreCard.reduce(0, +)
            DispatchQueue.main.async {
                self.state.text = "Score: \(Float(sum) / Float(scoreCard.count))"
                self.button.isEnabled = true
            }
        } catch {
            assertionFailure()
        }
    }
}

