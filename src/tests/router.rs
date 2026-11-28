use crate::config::Config;
use crate::http::Req;
use crate::router::handle;
use crate::testing::{block_on, FakePlatform};

fn req(method: &str, path: &str) -> Req {
    Req {
        method: method.into(),
        url: format!("https://api.shamwari.ai{path}"),
        headers: vec![],
        body: String::new(),
    }
}

#[test]
fn health_reports_the_service_version_and_mind() {
    let p = FakePlatform::new();
    let cfg = Config {
        mind_available: Some("true".into()),
        ..Default::default()
    };
    let r = block_on(handle(&p, &cfg, req("GET", "/health")));
    assert_eq!(r.status, 200);
    let b = r.body_json();
    assert_eq!(b["ok"], true);
    assert_eq!(b["service"], "shamwari-gateway");
    assert_eq!(b["mind"], true);
    assert_eq!(b["runtime"], "rust");
}

#[test]
fn mind_is_false_unless_exactly_true() {
    let p = FakePlatform::new();
    for v in [None, Some("TRUE"), Some("1"), Some("")] {
        let cfg = Config {
            mind_available: v.map(Into::into),
            ..Default::default()
        };
        let r = block_on(handle(&p, &cfg, req("GET", "/health")));
        assert_eq!(r.body_json()["mind"], false, "{v:?}");
    }
}

#[test]
fn an_unknown_route_or_method_is_not_found() {
    let p = FakePlatform::new();
    let cfg = Config::default();
    for (m, path) in [
        ("GET", "/"),
        ("GET", "/v1/chat/completions"),
        ("POST", "/v1/nope"),
        ("PUT", "/v1/models"),
    ] {
        let r = block_on(handle(&p, &cfg, req(m, path)));
        assert_eq!(r.status, 404, "{m} {path}");
        assert_eq!(r.body_json()["error"], "not_found");
    }
}
