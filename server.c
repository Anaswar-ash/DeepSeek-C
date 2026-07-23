#define DSV4_SERVER
#include "dsv4.c"
#include "mongoose.h"

static Model *g_model = NULL;
static struct mg_mgr g_server_mgr;

// Helper to escape JSON string for SSE
static void escape_json(const char *src, char *dst, size_t max_len) {
    size_t i = 0, j = 0;
    while (src[i] && j < max_len - 3) {
        if (src[i] == '"') { dst[j++] = '\\'; dst[j++] = '"'; }
        else if (src[i] == '\\') { dst[j++] = '\\'; dst[j++] = '\\'; }
        else if (src[i] == '\n') { dst[j++] = '\\'; dst[j++] = 'n'; }
        else if (src[i] == '\r') { dst[j++] = '\\'; dst[j++] = 'r'; }
        else if (src[i] == '\t') { dst[j++] = '\\'; dst[j++] = 't'; }
        else { dst[j++] = src[i]; }
        i++;
    }
    dst[j] = '\0';
}

static void send_chunk(struct mg_connection *c, const char *text) {
    if (!text || strlen(text) == 0) return;
    char escaped[512];
    escape_json(text, escaped, sizeof(escaped));
    mg_printf(c, "data: {\"choices\": [{\"delta\": {\"content\": \"%s\"}}]}\n\n", escaped);
    printf("Token generated: %s\n", text);
    fflush(stdout);
    mg_mgr_poll(&g_server_mgr, 0); // Flush the network buffer immediately
}

static void handle_chat_completion(struct mg_connection *c, struct mg_http_message *hm) {
    char prompt[4096] = {0};
    
    char *json_str = malloc(hm->body.len + 1);
    if (json_str) {
        memcpy(json_str, hm->body.buf, hm->body.len);
        json_str[hm->body.len] = '\0';
        char *arena = NULL;
        jval *root = json_parse(json_str, &arena);
        if (root) {
            jval *messages = json_get(root, "messages");
            if (messages && messages->t == J_ARR && messages->len > 0) {
                jval *last_msg = messages->kids[messages->len - 1];
                jval *content = json_get(last_msg, "content");
                if (content && content->t == J_STR) {
                    strncpy(prompt, content->str, sizeof(prompt) - 1);
                }
            }
            json_free(root);
        }
        if (arena) free(arena);
        free(json_str);
    }
    
    if (strlen(prompt) == 0) {
        strcpy(prompt, "Hello");
    }

    // Manually inject special tokens to avoid tokenization issues with full-width characters
    int input_ids[2048];
    input_ids[0] = 0;      // <｜begin of sentence｜>
    input_ids[1] = 128803; // <｜User｜>
    int n_tokens = 2;
    n_tokens += tok_encode(&g_model->tokenizer, prompt, strlen(prompt), input_ids + n_tokens, 2048 - 4);
    input_ids[n_tokens++] = 128804; // <｜Assistant｜>
    
    float temp = 0.7f;
    float top_p = 0.9f;

    mg_printf(c,
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/event-stream\r\n"
        "Cache-Control: no-cache\r\n"
        "Connection: keep-alive\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "\r\n");
    mg_mgr_poll(&g_server_mgr, 0);

    // Generation
    int pos = 0;
    for (int i = 0; i < g_model->c.n_layers + 1; i++) {
        memset(g_model->K[i], 0, (g_model->c.window_size + 4096) * g_model->c.kv_lora_rank * sizeof(float));
        memset(g_model->V[i], 0, (g_model->c.window_size + 4096) * g_model->c.kv_lora_rank * sizeof(float));
    }
    float *logits = malloc(g_model->c.vocab_size * sizeof(float));
    float *draft_logits = malloc(g_model->c.vocab_size * sizeof(float));
    float *main_hidden = malloc(g_model->c.hidden * sizeof(float));
    
    for(int i = 0; i < n_tokens; i++) {
        forward_dsv4(g_model, input_ids[i], pos++, logits, main_hidden);
    }
    
    int current_token = sample_topp(logits, g_model->c.vocab_size, temp, top_p);
    char buf[128];
    tok_decode(&g_model->tokenizer, &current_token, 1, buf, 128);
    send_chunk(c, buf);
    
    int draft_token = -1;
    
    for (int step = 0; step < 256; step++) {
        if (draft_token == -1 && current_token != 1) {
            // mtp_draft(g_model, current_token, main_hidden, pos, draft_logits);
            // draft_token = sample_topp(draft_logits, g_model->c.vocab_size, temp, top_p);
        }
        
        forward_dsv4(g_model, current_token, pos++, logits, main_hidden);
        int true_next = sample_topp(logits, g_model->c.vocab_size, temp, top_p);
        
        current_token = true_next;
        
        
        tok_decode(&g_model->tokenizer, &current_token, 1, buf, 128);
        printf("DEBUG: token_id=%d, piece='%s'\n", current_token, buf);
        if (true_next == 1) break; // EOS
        
        // Output sending
        if (c->is_closing) break;
        send_chunk(c, buf);
        draft_token = -1;
    }
    
    mg_printf(c, "data: [DONE]\n\n");
    c->is_draining = 1; // Tell Mongoose to close connection after sending
    mg_mgr_poll(&g_server_mgr, 0);
    
    free(logits);
    free(draft_logits);
    free(main_hidden);
}

static void ev_handler(struct mg_connection *c, int ev, void *ev_data) {
    if (ev == MG_EV_HTTP_MSG) {
        struct mg_http_message *hm = (struct mg_http_message *) ev_data;
        
        if (mg_match(hm->method, mg_str("OPTIONS"), NULL)) {
            mg_printf(c,
                "HTTP/1.1 204 No Content\r\n"
                "Access-Control-Allow-Origin: *\r\n"
                "Access-Control-Allow-Methods: POST, OPTIONS\r\n"
                "Access-Control-Allow-Headers: Content-Type, Authorization\r\n"
                "\r\n");
            return;
        }

        if (mg_match(hm->uri, mg_str("/v1/chat/completions"), NULL)) {
            handle_chat_completion(c, hm);
        } else {
            mg_http_reply(c, 404, "", "Not Found\n");
        }
    }
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc < 2) {
        printf("Usage: dsv4-server.exe <model_path>\n");
        return 1;
    }

    printf("==========================================\n");
    printf(" DeepSeek-C OpenAI-Compatible API Server  \n");
    printf("==========================================\n");
    
    g_model = calloc(1, sizeof(Model));
    model_init(g_model, argv[1]);
    
    printf("Model loaded. Starting server on http://localhost:8080\n");

    mg_mgr_init(&g_server_mgr);
    mg_http_listen(&g_server_mgr, "http://0.0.0.0:8080", ev_handler, NULL);
    
    while (1) {
        mg_mgr_poll(&g_server_mgr, 1000);
    }
    
    mg_mgr_free(&g_server_mgr);
    return 0;
}
