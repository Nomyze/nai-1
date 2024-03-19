use std::fs;
use std::io;
use std::collections::HashMap;
use std::io::Write;


#[derive(Clone)]
struct Rekord {
    attribs: Vec<f64>,
    class: String,
}

#[derive(Clone)]
struct TrainData {
    records: Vec<Rekord>,
}

fn main() {
    let train_data = from_file("iris_training.txt");
    println!("Displaying read data:");
    train_data.records.clone().into_iter().for_each(|rek| {
        println!("{}", rek.to_string());
    });
    let mut buffer = String::new();
    let stdin = io::stdin();
    print!("k: ");
    std::io::stdout().flush().unwrap();
    let _ = stdin.read_line(&mut buffer);
    let k = buffer.trim().parse::<i32>().expect("Failed to convert to an integer");
    let test = from_file("iris_test.txt");
    let mut count = 0;
    let mut hit = 0;
    for rek in test.records {
        let res = train_data.classify(k, &rek.attribs);
        println!("{:?}", &rek.attribs);
        println!("{}", &res);
        count += 1;
        if res.eq(&rek.class) {
            hit += 1;
        }
    }
    println!("Test count: {}\nHit count: {}\nPercentage: {}%", count, hit, hit as f64 / count as f64 * 100.0); 

    println!("Insert vector to classify: [attrb1 <whitespace> attrb2 <whitespace> ..] or 'q' to end");
    buffer = String::new();
    let _ = stdin.read_line(&mut buffer);
    while !buffer.trim().eq("q") {
        let rek = line_to_rekord(buffer.trim(), true);
        println!("{:?}", rek.attribs);
        println!("{}", train_data.classify(k, &rek.attribs));

        println!("Insert vector to classify: [attrb1 <whitespace> attrb2 <whitespace> ..] or 'q' to end");
        buffer = String::new();
        let _ = stdin.read_line(&mut buffer);
    }
}

fn from_file(path: &str) -> TrainData {
    let contents = fs::read_to_string(path)
        .expect("Failed to read file");
    let mut records: Vec<Rekord> = Vec::new();
    for line in contents.lines() {
        records.push(line_to_rekord(line, false));
    }
    TrainData { records }
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

impl TrainData {
    fn classify(&self, k: i32, in_vec: &Vec<f64>) -> String {
        let mut rip_mem: Vec<(f64, String)> = Vec::new();
        for rek in self.records.iter() {
            let dist = rek.distance(in_vec);
            rip_mem.push((dist, rek.class.clone()));
        }
        rip_mem.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let mut best: HashMap<&String, i32> = HashMap::new();
        for it in (0..k).zip(rip_mem.iter()) {
            *best.entry(&it.1.1).or_insert(0) += 1;
        }
        let mut ret: String = String::new();
        let mut max: i32 = 0;
        for key in best.keys() {
            if best[key] > max {
                max = best[key];
                ret = key.to_string();
            }
        }
        ret
    }
}

impl Rekord {
    fn distance(&self, other: &Vec<f64>) -> f64 {
        let mut sum = 0.;
        for it in self.attribs.iter().zip(other.iter()) {
            let (s, o) = it;
            sum += (s - o).powi(2);
        }
        //sum.sqrt() ommitted
        sum
    }
}

impl ToString for Rekord {
    fn to_string(&self) -> String {
        format!("Attributes: {:?}\tClassification: {}", self.attribs, self.class)
    }
}
