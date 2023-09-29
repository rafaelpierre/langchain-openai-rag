from mitmproxy import http

def request(flow: http.HTTPFlow):
    # redirect to different host
    if flow.request.pretty_host == "app.posthog.com":
        print(flow.request)
    # answer from proxy
    elif flow.request.path.endswith("/brew"):
    	flow.response = http.Response.make(
            418, b"I'm a teapot",
        )