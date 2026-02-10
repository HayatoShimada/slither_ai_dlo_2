"""
Selenium 経由で Chromium を制御し、slither.io を自動操作するモジュール。
Docker: Xvfb 上に実描画 (headed)。ローカル: headless=new でフォーカス奪取を防止。
"""

from __future__ import annotations

import os
import time

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from config import GAME_URL, NICKNAME, SCREEN_WIDTH, SCREEN_HEIGHT, HEADLESS_BROWSER


def create_driver() -> webdriver.Chrome:
    """
    Chromium WebDriver を生成して返す。

    Docker (Xvfb): headed モードで VNC に表示。
    ローカル: headless=new でフォーカス奪取を完全に防止。
    HEADLESS_BROWSER 設定で切替可能（デフォルト auto: Docker外なら headless）。

    Returns
    -------
    webdriver.Chrome
        設定済みの WebDriver インスタンス。
    """
    options = Options()
    options.binary_location = os.environ.get("CHROME_BIN", "/usr/bin/google-chrome-stable")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument(f"--window-size={SCREEN_WIDTH},{SCREEN_HEIGHT}")
    options.add_argument("--mute-audio")
    options.add_argument("--disable-gpu-sandbox")
    options.add_argument("--disable-extensions")

    # ウィンドウ最小化・非表示時の JS/レンダリング抑制を無効化
    options.add_argument("--disable-background-timer-throttling")
    options.add_argument("--disable-backgrounding-occluded-windows")
    options.add_argument("--disable-renderer-backgrounding")

    # ローカル実行: headless=new でブラウザウィンドウを非表示化
    # Chrome 112+ の headless=new は headed と同じレンダリングエンジンを使うため
    # canvas ゲームも正常動作し、スクリーンショットも取得可能
    if HEADLESS_BROWSER:
        options.add_argument("--headless=new")
        print("[Browser] Headless mode enabled (no visible window, no focus steal)")

    # Selenium 4.6+ は chromedriver を自動ダウンロード・管理する
    driver = webdriver.Chrome(options=options)

    # ページ読み込み前に anti-throttle JS を注入
    # Page.addScriptToEvaluateOnNewDocument はナビゲーション毎に自動再実行される
    _inject_anti_throttle(driver)

    return driver


def _inject_anti_throttle(driver: webdriver.Chrome) -> None:
    """
    ブラウザ最小化・非表示時にゲームが停止するのを防ぐ JS を注入する。

    対策:
      1. Page Visibility API をオーバーライド → ゲームが常に「表示中」と認識
      2. requestAnimationFrame を setTimeout(16ms) に置き換え →
         rAF はバックグラウンドで停止するが setTimeout は
         --disable-background-timer-throttling フラグで抑制解除済み
      3. CDP Page.addScriptToEvaluateOnNewDocument で注入するため
         ページ遷移・リロード後も自動的に再適用される
    """
    try:
        driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
            "source": """
                // --- Page Visibility API オーバーライド ---
                Object.defineProperty(document, 'hidden', {
                    get: function() { return false; }
                });
                Object.defineProperty(document, 'visibilityState', {
                    get: function() { return 'visible'; }
                });
                document.addEventListener('visibilitychange', function(e) {
                    e.stopImmediatePropagation();
                }, true);

                // --- requestAnimationFrame → setTimeout 置き換え ---
                // rAF はウィンドウ非表示時にブラウザが停止させる。
                // setTimeout(16ms ≈ 60fps) なら Chrome フラグで抑制解除済み。
                window.requestAnimationFrame = function(cb) {
                    return window.setTimeout(function() { cb(performance.now()); }, 16);
                };
                window.cancelAnimationFrame = function(id) {
                    window.clearTimeout(id);
                };
            """
        })
    except Exception as e:
        print(f"WARNING: Failed to inject anti-throttle script: {e}")


def start_game(driver: webdriver.Chrome) -> None:
    """
    slither.io に遷移し、ニックネーム入力 → Play ボタンクリックでゲームを開始する。

    Parameters
    ----------
    driver : webdriver.Chrome
        create_driver() で生成した WebDriver。
    """
    driver.get(GAME_URL)

    # ページ読み込み待機
    WebDriverWait(driver, 30).until(
        EC.presence_of_element_located((By.ID, "nick"))
    )
    time.sleep(1)  # アセット読み込み余裕

    # ニックネーム入力
    nick_input = driver.find_element(By.ID, "nick")
    nick_input.clear()
    nick_input.send_keys(NICKNAME)

    # Play ボタンクリック (CSS セレクタ → JS フォールバック)
    try:
        play_btn = driver.find_element(By.CSS_SELECTOR, ".play-btn, .btn-play, #playh")
        play_btn.click()
    except Exception:
        driver.execute_script(
            "var btn = document.querySelector('.play-btn') || "
            "document.querySelector('.btn-play') || "
            "document.getElementById('playh'); "
            "if(btn) btn.click(); else if(window.play) window.play();"
        )

    # ゲーム開始を待機
    for _ in range(60):
        if is_playing(driver):
            print("Game started successfully.")
            return
        time.sleep(0.5)

    print("WARNING: Game start not confirmed, proceeding anyway.")


def is_playing(driver: webdriver.Chrome) -> bool:
    """
    ゲームがプレイ中かどうかを JS で確認する。

    Returns
    -------
    bool
        プレイ中なら True。
    """
    try:
        return driver.execute_script("return window.playing || false;")
    except Exception:
        return False


def is_game_over(driver: webdriver.Chrome) -> bool:
    """
    ゲームオーバーかどうかを判定する。

    Returns
    -------
    bool
        ゲームオーバーなら True。
    """
    return not is_playing(driver)


def restart_game(driver: webdriver.Chrome) -> None:
    """
    ゲームオーバー後にページをフルリロードして再プレイする。
    ボタンクリックは不安定なため、常にフルリロードで確実にリスタートする。
    """
    print("Restarting game (full reload)...")
    start_game(driver)


def get_game_state(driver: webdriver.Chrome) -> dict:
    """
    1回の JS 呼び出しでスコア・マップ位置・境界比率をまとめて取得する。
    Selenium のラウンドトリップを最小化するためバッチ化。

    Returns
    -------
    dict
        {"score": int, "boundary_ratio": float, "map_dx": float,
         "map_dy": float, "playing": bool}
        boundary_ratio: 0.0=中心, 1.0=境界。-1.0=取得失敗。
        map_dx, map_dy: マップ中心からの正規化座標 (-1.0〜+1.0)。
        取得失敗時はデフォルト値。
    """
    try:
        result = driver.execute_script("""
            var out = {score: 0, boundary_ratio: -1, map_dx: 0, map_dy: 0,
                        playing: false, _debug: ''};
            out.playing = !!(window.playing);

            // --- snake オブジェクトを探す ---
            var s = window.snake || null;

            // フォールバック1: snakes 配列の先頭
            if (!s && window.snakes && Array.isArray(window.snakes)) {
                for (var i = 0; i < window.snakes.length; i++) {
                    if (window.snakes[i]) { s = window.snakes[i]; out._debug = 'snakes['+i+']'; break; }
                }
            }

            // フォールバック2: 全グローバル変数から pts 配列を持つオブジェクトを探す
            if (!s) {
                var checked = 0;
                for (var gk in window) {
                    checked++;
                    if (checked > 8000) break;
                    try {
                        var gv = window[gk];
                        if (!gv || typeof gv !== 'object') continue;
                        // 配列の中にptsを持つ要素がある → snakes配列
                        if (Array.isArray(gv) && gv.length > 0 && gv.length < 500) {
                            for (var ai = 0; ai < gv.length; ai++) {
                                var elem = gv[ai];
                                if (elem && typeof elem === 'object'
                                    && elem.pts && Array.isArray(elem.pts)
                                    && elem.pts.length > 0) {
                                    s = elem;
                                    out._debug = 'found:' + gk + '[' + ai + ']';
                                    break;
                                }
                            }
                            if (s) break;
                        }
                        // 単体オブジェクトにptsがある
                        if (!Array.isArray(gv) && gv.pts && Array.isArray(gv.pts)
                            && gv.pts.length > 0) {
                            s = gv;
                            out._debug = 'found:' + gk;
                            break;
                        }
                    } catch(e) {}
                }
            }
            if (!s) { out._debug = 'no_snake(scanned:' + (checked||0) + ')'; return out; }

            // --- スコア（安定した指標を優先） ---
            // 実スコア計算: sct + fam + fpsls/fmlts テーブル
            if (typeof s.sct === 'number' && s.sct > 0) {
                try {
                    var sc = Math.min(6, s.sct);
                    var fp = (window.fpsls && window.fpsls[sc]) || 0;
                    var fm = (window.fmlts && window.fmlts[sc]) || 1;
                    var fam = (typeof s.fam === 'number') ? s.fam : 0;
                    var realScore = Math.floor(15 * (fp + fam / fm - 1) - 5);
                    out.score = (realScore > 0) ? realScore : s.sct;
                } catch(e) {
                    out.score = s.sct;
                }
            }
            // フォールバック: fam → pts.length
            if (out.score <= 0 && typeof s.fam === 'number' && s.fam > 0)
                out.score = Math.floor(s.fam);
            if (out.score <= 0 && s.pts && s.pts.length > 0)
                out.score = Math.floor(s.pts.length / 3);

            // --- マップ位置（複数パターンで探索） ---
            var x = NaN, y = NaN, src = '';

            // パターン1: 直接プロパティ (xx/yy, x/y, gx/gy)
            if (typeof s.xx === 'number') { x = s.xx; y = s.yy; src = 'snake.xx'; }
            if (isNaN(x) && typeof s.x === 'number' && s.x > 100) {
                x = s.x; y = s.y; src = 'snake.x';
            }
            if (isNaN(x) && typeof s.gx === 'number') {
                x = s.gx; y = s.gy; src = 'snake.gx';
            }

            // パターン2: pts 配列（末尾 = head が一般的、先頭もチェック）
            if (isNaN(x) && s.pts && s.pts.length > 0) {
                var ends = [
                    {el: s.pts[s.pts.length - 1], tag: 'pts_last'},
                    {el: s.pts[0],                 tag: 'pts_first'},
                ];
                for (var i = 0; i < ends.length; i++) {
                    var p = ends[i].el;
                    if (!p) continue;
                    if (typeof p.xx === 'number' && p.xx > 100) {
                        x = p.xx; y = p.yy; src = ends[i].tag + '.xx'; break;
                    }
                    if (typeof p.x === 'number' && p.x > 100) {
                        x = p.x; y = p.y; src = ends[i].tag + '.x'; break;
                    }
                }
            }

            // パターン3: chl (head element 参照)
            if (isNaN(x) && s.chl) {
                if (typeof s.chl.xx === 'number') { x = s.chl.xx; y = s.chl.yy; src = 'chl.xx'; }
                else if (typeof s.chl.x === 'number' && s.chl.x > 100) {
                    x = s.chl.x; y = s.chl.y; src = 'chl.x';
                }
            }

            // パターン4: ヒューリスティック（座標らしい数値プロパティを探索）
            if (isNaN(x)) {
                var cands = [];
                for (var k in s) {
                    try {
                        if (typeof s[k] === 'number' && s[k] > 1000 && s[k] < 65000)
                            cands.push({k: k, v: s[k]});
                    } catch(e) {}
                }
                // 値が近い2つのペアを座標と推定（マップ中心 ~21600 付近）
                if (cands.length >= 2) {
                    cands.sort(function(a,b) { return Math.abs(a.v - 21600) - Math.abs(b.v - 21600); });
                    x = cands[0].v; y = cands[1].v;
                    src = 'heuristic:' + cands[0].k + '+' + cands[1].k;
                }
            }

            if (src) out._debug = src;
            else out._debug = 'pos_not_found';

            // --- 境界比率の計算 ---
            var grd = 21600;
            if (typeof window.grd === 'number' && window.grd > 0) grd = window.grd;
            else if (typeof window.gsc === 'number' && window.gsc > 0) grd = window.gsc * 24000;

            if (grd > 0 && !isNaN(x) && !isNaN(y)) {
                var dx = x - grd;
                var dy = y - grd;
                var dist = Math.sqrt(dx*dx + dy*dy);
                out.boundary_ratio = Math.min(dist / grd, 1.0);
                out.map_dx = Math.max(-1.0, Math.min(1.0, dx / grd));
                out.map_dy = Math.max(-1.0, Math.min(1.0, dy / grd));
            }
            return out;
        """)
        if result and isinstance(result, dict):
            return {
                "score": int(result.get("score", 0)),
                "boundary_ratio": float(result.get("boundary_ratio", -1.0)),
                "map_dx": float(result.get("map_dx", 0.0)),
                "map_dy": float(result.get("map_dy", 0.0)),
                "playing": bool(result.get("playing", False)),
                "_debug": str(result.get("_debug", "")),
            }
    except Exception:
        pass
    return {"score": 0, "boundary_ratio": -1.0, "map_dx": 0.0, "map_dy": 0.0,
            "playing": False, "_debug": "exception"}


def get_map_boundary_ratio(driver: webdriver.Chrome) -> float:
    """後方互換ラッパー。"""
    return get_game_state(driver)["boundary_ratio"]


def dump_snake_properties(driver: webdriver.Chrome) -> None:
    """
    window.snake 周辺のゲーム内部状態を詳細にログ出力する（診断用）。

    snake オブジェクトが見つからない場合は、グローバル変数から
    snake 風のオブジェクトを探索する。
    """
    try:
        result = driver.execute_script("""
            var diag = {};
            diag.playing = window.playing;
            diag.snake_exists = (window.snake !== undefined && window.snake !== null);
            diag.grd = window.grd;
            diag.gsc = window.gsc;

            // --- window.snake が存在する場合 ---
            if (window.snake) {
                var s = window.snake;
                var nums = {}, strs = {}, bools = {}, arrays = {}, objs = {};
                for (var k in s) {
                    try {
                        var v = s[k];
                        var t = typeof v;
                        if (t === 'number')       nums[k] = v;
                        else if (t === 'string')   strs[k] = v.substring(0, 50);
                        else if (t === 'boolean')  bools[k] = v;
                        else if (Array.isArray(v)) arrays[k] = v.length;
                        else if (v === null)        nums[k] = null;
                        else if (t === 'object')   objs[k] = Object.keys(v).slice(0, 8).join(',');
                        // function は無視
                    } catch(e) {}
                }
                diag.snake_numeric = nums;
                diag.snake_string = strs;
                diag.snake_boolean = bools;
                diag.snake_arrays = arrays;
                diag.snake_objects = objs;

                // pts 配列の中身（先頭・末尾の数値プロパティ）
                if (s.pts && s.pts.length > 0) {
                    diag.pts_length = s.pts.length;
                    var examine = function(elem, label) {
                        if (!elem) return;
                        var p = {};
                        for (var k in elem) {
                            if (typeof elem[k] === 'number') p[k] = elem[k];
                            else if (typeof elem[k] === 'object' && elem[k] !== null)
                                p[k] = '{' + Object.keys(elem[k]).slice(0,5).join(',') + '}';
                        }
                        diag[label] = p;
                    };
                    examine(s.pts[0], 'pts_first');
                    examine(s.pts[s.pts.length - 1], 'pts_last');
                }
            }

            // --- snake 風のグローバル変数を全探索 ---
            if (!window.snake) {
                // 全 window プロパティを走査し、pts 配列を持つオブジェクトを探す
                var pts_objects = {};
                var coord_objects = {};
                var checked = 0;
                for (var gk in window) {
                    checked++;
                    if (checked > 8000) break;
                    try {
                        var gv = window[gk];
                        if (!gv || typeof gv !== 'object') continue;

                        // 配列: 中身に pts を持つ要素があるか
                        if (Array.isArray(gv) && gv.length > 0 && gv.length < 500) {
                            for (var ai = 0; ai < Math.min(gv.length, 5); ai++) {
                                var elem = gv[ai];
                                if (elem && typeof elem === 'object' && elem.pts
                                    && Array.isArray(elem.pts)) {
                                    pts_objects[gk] = {
                                        type: 'Array(' + gv.length + ')',
                                        idx: ai,
                                        pts_len: elem.pts.length,
                                        elem_keys: Object.keys(elem).slice(0, 15).join(',')
                                    };
                                    break;
                                }
                            }
                        }

                        // 単体オブジェクト: pts を持つか
                        if (!Array.isArray(gv) && gv.pts && Array.isArray(gv.pts)) {
                            pts_objects[gk] = {
                                type: 'Object',
                                pts_len: gv.pts.length,
                                keys: Object.keys(gv).slice(0, 15).join(',')
                            };
                        }

                        // 座標っぽい数値プロパティ (grd=32550 の ±50% 範囲)
                        if (!Array.isArray(gv) && typeof gv === 'object') {
                            var cn = 0;
                            var cp = {};
                            for (var ck in gv) {
                                if (typeof gv[ck] === 'number'
                                    && gv[ck] > 10000 && gv[ck] < 65000) {
                                    cp[ck] = gv[ck];
                                    cn++;
                                    if (cn >= 4) break;
                                }
                            }
                            if (cn >= 2) coord_objects[gk] = cp;
                        }
                    } catch(e) {}
                }
                diag.checked_globals = checked;
                diag.pts_objects = pts_objects;
                diag.coord_objects = coord_objects;
            }

            // --- snakes 配列チェック ---
            if (window.snakes && Array.isArray(window.snakes)) {
                diag.snakes_length = window.snakes.length;
                // 最初の非null要素の数値プロパティ
                for (var i = 0; i < Math.min(window.snakes.length, 3); i++) {
                    var sn = window.snakes[i];
                    if (sn) {
                        var p = {};
                        for (var k in sn) {
                            if (typeof sn[k] === 'number') p[k] = sn[k];
                            else if (Array.isArray(sn[k])) p[k] = 'Array(' + sn[k].length + ')';
                        }
                        diag['snakes_' + i] = p;
                        break;
                    }
                }
            }

            return JSON.stringify(diag, null, 2);
        """)
        print(f"[DEBUG] Game internals dump:\n{result}", flush=True)
    except Exception as e:
        print(f"[DEBUG] Failed to dump game internals: {e}", flush=True)


def get_score(driver: webdriver.Chrome) -> int:
    """後方互換ラッパー。"""
    return get_game_state(driver)["score"]


def get_js_entities(driver: webdriver.Chrome) -> dict:
    """
    JS injection で敵スネーク・食物の座標をまとめて取得する。

    ゲーム内部の window.snakes / foods 配列から直接座標を読み取り、
    プレイヤーからの相対スクリーン座標に変換して返す。
    ビジョンベース検出よりも高精度・高速。

    Returns
    -------
    dict
        {
            "enemies": list[dict]  -- 各敵: {sx, sy, heading, length}
                                      sx/sy はスクリーンピクセル座標
            "foods": list[list[int]]  -- 各食物: [sx, sy]
            "gsc": float  -- ゲームのズームスケール
            "ok": bool  -- 取得成功フラグ
        }
    """
    try:
        result = driver.execute_script("""
            var out = {enemies: [], foods: [], gsc: 0, ok: false};

            // プレイヤースネークの取得
            var me = window.snake || null;
            if (!me && window.snakes && Array.isArray(window.snakes)) {
                for (var i = 0; i < window.snakes.length; i++) {
                    if (window.snakes[i]) { me = window.snakes[i]; break; }
                }
            }
            if (!me) return out;

            // プレイヤーの座標
            var mx = NaN, my = NaN;
            if (typeof me.xx === 'number') { mx = me.xx; my = me.yy; }
            else if (typeof me.x === 'number' && me.x > 100) { mx = me.x; my = me.y; }
            if (isNaN(mx)) return out;

            // ズームスケール
            var gsc = (typeof window.gsc === 'number' && window.gsc > 0) ? window.gsc : 0.9;
            out.gsc = gsc;

            // スクリーン中心
            var scx = """ + str(SCREEN_WIDTH // 2) + """;
            var scy = """ + str(SCREEN_HEIGHT // 2) + """;

            // --- 敵スネーク ---
            if (window.snakes && Array.isArray(window.snakes)) {
                var maxEnemy = 20;  // 上位20体まで
                for (var i = 0; i < window.snakes.length && out.enemies.length < maxEnemy; i++) {
                    var s = window.snakes[i];
                    if (!s || s === me) continue;

                    var ex = NaN, ey = NaN;
                    if (typeof s.xx === 'number') { ex = s.xx; ey = s.yy; }
                    else if (typeof s.x === 'number' && s.x > 100) { ex = s.x; ey = s.y; }
                    if (isNaN(ex)) continue;

                    // ワールド座標差をスクリーンピクセルに変換
                    var sx = (ex - mx) * gsc + scx;
                    var sy = (ey - my) * gsc + scy;

                    // 画面外すぎるものはスキップ（画面の2倍まで）
                    if (sx < -scx || sx > scx * 3 || sy < -scy || sy > scy * 3) continue;

                    // heading (ang プロパティ)
                    var heading = 0;
                    if (typeof s.ang === 'number') heading = s.ang;
                    else if (typeof s.eang === 'number') heading = s.eang;

                    // 長さ推定 (sct = segment count)
                    var length = 0;
                    if (typeof s.sct === 'number') length = s.sct;
                    else if (s.pts && Array.isArray(s.pts)) length = s.pts.length;

                    out.enemies.push({sx: Math.round(sx), sy: Math.round(sy),
                                      heading: heading, length: length});
                }
            }

            // --- 食物 ---
            // slither.io の食物は foods 配列またはグローバル配列に格納
            var foodArr = window.foods || window.pellets || null;
            if (!foodArr && window.foods_c) foodArr = window.foods_c;
            if (foodArr && Array.isArray(foodArr)) {
                var maxFood = 50;
                for (var i = 0; i < foodArr.length && out.foods.length < maxFood; i++) {
                    var f = foodArr[i];
                    if (!f) continue;

                    var fx = NaN, fy = NaN;
                    if (typeof f.xx === 'number') { fx = f.xx; fy = f.yy; }
                    else if (typeof f.x === 'number') { fx = f.x; fy = f.y; }
                    if (isNaN(fx)) continue;

                    var fsx = (fx - mx) * gsc + scx;
                    var fsy = (fy - my) * gsc + scy;

                    // 画面内のみ
                    if (fsx < 0 || fsx > scx * 2 || fsy < 0 || fsy > scy * 2) continue;

                    out.foods.push([Math.round(fsx), Math.round(fsy)]);
                }
            }

            out.ok = true;
            return out;
        """)
        if result and isinstance(result, dict) and result.get("ok"):
            return result
    except Exception:
        pass
    return {"enemies": [], "foods": [], "gsc": 0, "ok": False}


def inject_toggle_listener(driver: webdriver.Chrome) -> None:
    """Tab キーで human_mode を切り替える JS リスナーをブラウザに注入する。

    ページリロード後は再注入が必要。
    """
    try:
        driver.execute_script("""
            if (!window._toggle_listener_injected) {
                window._human_mode = false;
                document.addEventListener('keydown', function(e) {
                    if (e.key === 'Tab') {
                        e.preventDefault();
                        window._human_mode = !window._human_mode;
                        console.log('[Toggle] human_mode=' + window._human_mode);
                    }
                }, true);
                window._toggle_listener_injected = true;
            }
        """)
    except Exception:
        pass


def is_human_mode(driver: webdriver.Chrome) -> bool:
    """現在人間モードかどうかを JS から取得する。"""
    try:
        return bool(driver.execute_script("return window._human_mode || false;"))
    except Exception:
        return False
