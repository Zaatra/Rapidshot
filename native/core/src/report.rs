//! An ordered set of named readings, the shape the diagnostic probes report in.
//!
//! The probes used to write straight into a Python `dict` as they went. That
//! tied measurement code to the interpreter, so it now fills a `Report`, and
//! the binding turns that into the same `dict`, in the same key order, with the
//! same values -- a missing reading is `None` there, as it was.

/// One reading.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    /// Not known: `None` in Python. `Option<bool>` readings that are `None`.
    Missing,
    List(Vec<Report>),
}

/// Named readings in the order they were first set. Setting a key again
/// replaces its value in place, as assigning to a `dict` key does.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Report {
    pub fields: Vec<(String, Value)>,
}

impl Report {
    pub fn set(&mut self, key: &str, value: impl Into<Value>) {
        let value = value.into();
        match self.fields.iter_mut().find(|(k, _)| k == key) {
            Some(slot) => slot.1 = value,
            None => self.fields.push((key.to_owned(), value)),
        }
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        self.fields.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }
}

impl From<bool> for Value {
    fn from(v: bool) -> Self {
        Value::Bool(v)
    }
}

impl From<u32> for Value {
    fn from(v: u32) -> Self {
        Value::Int(i64::from(v))
    }
}

impl From<u64> for Value {
    fn from(v: u64) -> Self {
        Value::Int(i64::try_from(v).unwrap_or(i64::MAX))
    }
}

impl From<usize> for Value {
    fn from(v: usize) -> Self {
        Value::Int(i64::try_from(v).unwrap_or(i64::MAX))
    }
}

impl From<f64> for Value {
    fn from(v: f64) -> Self {
        Value::Float(v)
    }
}

impl From<&str> for Value {
    fn from(v: &str) -> Self {
        Value::Str(v.to_owned())
    }
}

impl From<String> for Value {
    fn from(v: String) -> Self {
        Value::Str(v)
    }
}

impl From<&String> for Value {
    fn from(v: &String) -> Self {
        Value::Str(v.clone())
    }
}

impl From<Option<bool>> for Value {
    fn from(v: Option<bool>) -> Self {
        v.map_or(Value::Missing, Value::Bool)
    }
}

impl From<Vec<Report>> for Value {
    fn from(v: Vec<Report>) -> Self {
        Value::List(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_insertion_order_and_replaces_in_place() {
        let mut r = Report::default();
        r.set("a", true);
        r.set("b", 2u32);
        r.set("a", false);
        assert_eq!(
            r.fields,
            vec![
                ("a".to_owned(), Value::Bool(false)),
                ("b".to_owned(), Value::Int(2))
            ]
        );
    }

    #[test]
    fn an_unknown_reading_is_missing_not_false() {
        let mut r = Report::default();
        r.set("x", None::<bool>);
        assert_eq!(r.get("x"), Some(&Value::Missing));
    }
}
