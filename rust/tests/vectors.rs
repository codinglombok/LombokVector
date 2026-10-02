//! Runs the shared vectors (vectors/lombokvector-vectors-v1.json): every
//! expected IEEE 754 bit pattern must match exactly, binary64 and binary32.

use lombokvector as lv;
use serde_json::{json, Value};

fn f64s(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

fn f32s(v: &Value) -> Vec<f32> {
    // binary32 inputs are written as the exact decimal of the binary32 value
    f64s(v).into_iter().map(|x| x as f32).collect()
}

fn b64(x: f64) -> Value {
    json!(format!("0x{:016x}", x.to_bits()))
}

fn b32(x: f32) -> Value {
    json!(format!("0x{:08x}", x.to_bits()))
}

fn wrap<T>(r: Result<T, lv::VectorError>, enc: impl Fn(T) -> Value) -> Value {
    match r {
        Ok(v) => json!({ "value": enc(v) }),
        Err(e) => json!({ "error": e.code() }),
    }
}

macro_rules! runner {
    ($name:ident, $t:ty, $conv:ident, $bits:ident, $dot:path, $norm:path, $l2:path, $cos:path,
     $nrm:path, $add:path, $sub:path, $scale:path, $bcos:path, $bdot:path, $bl2:path, $mat:path) => {
        fn $name(op: &str, args: &[Value]) -> Value {
            let list = |v: Vec<$t>| Value::Array(v.into_iter().map($bits).collect());
            let ranked = |v: Vec<(usize, $t)>| {
                Value::Array(v.into_iter().map(|(i, s)| json!([i, $bits(s)])).collect())
            };
            let set =
                |v: &Value| -> Vec<Vec<$t>> { v.as_array().unwrap().iter().map($conv).collect() };
            match op {
                "dot" => wrap($dot(&$conv(&args[0]), &$conv(&args[1])), $bits),
                "norm" => wrap($norm(&$conv(&args[0])), $bits),
                "l2" => wrap($l2(&$conv(&args[0]), &$conv(&args[1])), $bits),
                "cosine" => wrap($cos(&$conv(&args[0]), &$conv(&args[1])), $bits),
                "normalize" => wrap($nrm(&$conv(&args[0])), list),
                "add" => wrap($add(&$conv(&args[0]), &$conv(&args[1])), list),
                "sub" => wrap($sub(&$conv(&args[0]), &$conv(&args[1])), list),
                "scale" => wrap(
                    $scale(&$conv(&args[0]), args[1].as_f64().unwrap() as $t),
                    list,
                ),
                "batch_cosine" | "batch_dot" | "batch_l2" => {
                    let q = $conv(&args[0]);
                    let c = set(&args[1]);
                    let refs: Vec<&[$t]> = c.iter().map(|v| v.as_slice()).collect();
                    let r = match op {
                        "batch_cosine" => $bcos(&q, &refs),
                        "batch_dot" => $bdot(&q, &refs),
                        _ => $bl2(&q, &refs),
                    };
                    wrap(r, ranked)
                }
                "matrix_cosine" => {
                    let (a, b) = (set(&args[0]), set(&args[1]));
                    let ra: Vec<&[$t]> = a.iter().map(|v| v.as_slice()).collect();
                    let rb: Vec<&[$t]> = b.iter().map(|v| v.as_slice()).collect();
                    wrap($mat(&ra, &rb), |m: Vec<Vec<$t>>| {
                        Value::Array(m.into_iter().map(list).collect())
                    })
                }
                other => panic!("unknown op {other}"),
            }
        }
    };
}

runner!(
    run64,
    f64,
    f64s,
    b64,
    lv::dot_product_f64,
    lv::l2_norm_f64,
    lv::l2_distance_f64,
    lv::cosine_similarity_f64,
    lv::normalize_f64,
    lv::vec_add_f64,
    lv::vec_sub_f64,
    lv::vec_mul_scalar_f64,
    lv::batch_cosine_f64,
    lv::batch_dot_f64,
    lv::batch_l2_f64,
    lv::distance_matrix_cosine_f64
);
runner!(
    run32,
    f32,
    f32s,
    b32,
    lv::dot_product,
    lv::l2_norm,
    lv::l2_distance,
    lv::cosine_similarity,
    lv::normalize,
    lv::vec_add,
    lv::vec_sub,
    lv::vec_mul_scalar,
    lv::batch_cosine,
    lv::batch_dot,
    lv::batch_l2,
    lv::distance_matrix_cosine
);

#[test]
fn vectors() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../vectors/lombokvector-vectors-v1.json"
    );
    let doc: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let cases = doc["cases"].as_array().unwrap();
    assert!(cases.len() >= 100);
    let mut failures = Vec::new();
    for c in cases {
        let args = c["args"].as_array().unwrap();
        let op = c["op"].as_str().unwrap();
        let got = match c["precision"].as_str().unwrap() {
            "f64" => run64(op, args),
            _ => run32(op, args),
        };
        if got != c["expected"] {
            failures.push(format!(
                "{}: got {} expected {}",
                c["id"], got, c["expected"]
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{} failures (backend {}):\n{}",
        failures.len(),
        lv::active_backend(),
        failures.join("\n")
    );
}
