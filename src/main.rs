use std::fs;
use std::io;
use std::io::Write;
use rand::thread_rng;
use rand::seq::SliceRandom;

#[derive(Clone)]
struct Rekord {
    attribs: Vec<f64>,
    class: String,
}

struct Perceptron {
    weights: Vec<f64>,
    theta: f64,
}

fn main() {
    let mut buffer = String::new();
    let stdin = io::stdin();
    print!("alpha: ");
    std::io::stdout().flush().unwrap();
    let _ = stdin.read_line(&mut buffer);
    let k = buffer.trim().parse::<f64>().expect("Failed to convert to a float");
    let perceptron = from_file("iris_training.txt", k);
    println!("Displaying read data:");
    println!("w: {:?} t: {}", perceptron.weights, perceptron.theta);
    //let test = from_file("iris_test.txt");
    let contents = fs::read_to_string("iris_test.txt")
        .expect("Failed to read file");
    let mut records: Vec<Rekord> = Vec::new();
    for line in contents.lines() {
        records.push(line_to_rekord(line, false));
    }
    
    let mut count = 0;
    let mut hit = 0;
    for rek in records {
        //let res = train_data.classify(k, &rek.attribs);
        let out = perceptron.get_output(&rek);
        let res = if out == 1 {"Iris-setosa"} else {"Not Iris-setosa"};

        println!("Test data: {:?} {}", &rek.attribs, &rek.class);
        println!("Classified as: {}", &res);
        count += 1;
        if (out == 1 && res.eq(&rek.class)) || (out == 0 && rek.class.ne("Iris-setosa")) {
            hit += 1;
        }
    }
    println!("Test count: {}\nHit count: {}\nPercentage: {:.2}%", count, hit, hit as f64 / count as f64 * 100.0); 
    println!("Insert vector to classify: [attrb1 <whitespace> attrb2 <whitespace> ..] or 'q' to end");
    buffer.clear();
    let _ = stdin.read_line(&mut buffer);
    while !buffer.trim().eq("q") {
        let rek = line_to_rekord(buffer.trim(), true);
        let out = perceptron.get_output(&rek);
        let res = if out == 1 {"Iris-setosa"} else {"Not Iris-setosa"};
        println!("Testing: {:?}", &rek.attribs);
        println!("Classified as: {}", res);

        println!("Insert vector to classify: [attrb1 <whitespace> attrb2 <whitespace> ..] or 'q' to end");
        buffer.clear();
        let _ = stdin.read_line(&mut buffer);
    }
}

fn from_file(path: &str, alpha: f64) -> Perceptron {
    let contents = fs::read_to_string(path)
        .expect("Failed to read file");
    let mut records: Vec<Rekord> = Vec::new();
    for line in contents.lines() {
        records.push(line_to_rekord(line, false));
    }
    let mut perc = Perceptron {
        weights: vec![0.0; records[0].attribs.len()],
        theta: 0.0,
    };
    records.shuffle(&mut thread_rng());
    for rek in records {
        let d = if rek.class == "Iris-setosa" {1} else {0};
        let y = perc.get_output(&rek);
        for i in 0..perc.weights.len() {
            perc.weights[i] += (d - y) as f64 * alpha * rek.attribs.clone()[i];
        }
        perc.theta += (d - y) as f64 * alpha * -1.0;
    }
    perc
}

fn line_to_rekord(line: &str, noclass: bool) -> Rekord {
    let mut columns = line.split_whitespace();
    let mut atrs: Vec<f64> = Vec::new();
    let mut cls: String = String::new();
    if !noclass {
        cls = columns.next_back().unwrap().to_string();
    }
    for atr in columns {
        let parsed = match atr.trim().replace(',', ".").parse::<f64>() {
            Ok(x) => x,
            Err(_) => {
                eprintln!(r"Error parsing `{}` to f64 falling back to 0.0f", atr);
                0.
            },
        };

        atrs.push(parsed);
    }
    Rekord {
        attribs: atrs,
        class: cls,
    }
}
impl Perceptron {
    fn get_output(&self, rek: &Rekord) -> i32 {
        let mut w_val = 0.0;
        for z_iter in self.weights.iter().zip(rek.attribs.iter()) {
            w_val += z_iter.0 * z_iter.1;
        }
        //println!("perc: {:?}\nvec: {:?}\nval: {} theta: {}", self.weights, rek.attribs, w_val, self.theta);
        if w_val >= self.theta {1} else {0}
    }
}

impl ToString for Rekord {
    fn to_string(&self) -> String {
        format!("Attributes: {:?}\tClassification: {}", self.attribs, self.class)
    }
}
