//! The request and response the router works with, independent of the runtime.

use serde_json::Value;

#[derive(Debug, Clone, Default)]
pub struct Req {
    pub method: String,
    pub url: String,
    /// Lower-cased names.
    pub headers: Vec<(String, String)>,
    pub body: String,
}

impl Req {
    pub fn header(&self, name: &str) -> Option<&str> {
        let name = name.to_ascii_lowercase();
        self.headers
            .iter()
            .find(|(k, _)| *k == name)
            .map(|(_, v)| v.as_str())
    }

    pub fn path(&self) -> String {
        url::Url::parse(&self.url)
            .map(|u| u.path().to_owned())
            .unwrap_or_else(|_| "/".to_owned())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Resp {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: String,
}

impl Resp {
    /// A JSON response, as the TypeScript gateway's `json(body, status)`:
    /// `JSON.stringify(body, null, 2)`, two-space indentation, which is what
    /// `serde_json::to_string_pretty` writes byte for byte.
    pub fn json(body: &Value, status: u16) -> Self {
        Self::with_json_body(
            serde_json::to_string_pretty(body).unwrap_or_else(|_| body.to_string()),
            status,
        )
    }

    /// A JSON response written as `JSON.stringify(body)` with no indentation:
    /// what `responses.ts`, `channels/channel.ts` and the compact envelope send.
    pub fn json_compact(body: &Value, status: u16) -> Self {
        Self::with_json_body(body.to_string(), status)
    }

    fn with_json_body(body: String, status: u16) -> Self {
        Resp {
            status,
            headers: vec![("content-type".into(), "application/json".into())],
            body,
        }
    }

    /// Statuses whose response has no body (`new Response(null, ...)`): the
    /// runtime refuses a body with them.
    pub fn is_null_body_status(status: u16) -> bool {
        matches!(status, 101 | 204 | 205 | 304)
    }

    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }

    pub fn set_header(&mut self, name: &str, value: &str) {
        self.headers.retain(|(k, _)| !k.eq_ignore_ascii_case(name));
        self.headers
            .push((name.to_ascii_lowercase(), value.to_owned()));
    }

    pub fn body_json(&self) -> Value {
        serde_json::from_str(&self.body).unwrap_or(Value::Null)
    }
}
