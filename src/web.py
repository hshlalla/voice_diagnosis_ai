from waitress import serve
from flask import Flask, request, render_template, jsonify, url_for, session
import os
import configure, diagnosis

app = Flask(__name__)
app.secret_key = 'gkdlekql3#'  # Secret Key

logger = configure.getLogger('web')

sw_version = 'Release 0.0.006' 

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/test_result')
def test_result():
    """결과 페이지"""
    diagnosis_result = session.get('diagnosis_result', {})  
    return render_template('test_result.html', result=diagnosis_result)

@app.route('/diagnosis', methods=['POST'])
def request_diagnosis():
    """진단 API"""
    uploaded_files = request.files.getlist("fileList")
    save_dir = "/tmp/voice_files"
    os.makedirs(save_dir, exist_ok=True)

    file_paths = []
    for file in uploaded_files:
        save_path = os.path.join(save_dir, file.filename)
        file.save(save_path)
        file_paths.append(save_path)

    result_data = diagnosis.execute(file_paths)  # 진단 결과

    # 🔹 세션을 이용하여 데이터를 안전하게 전달
    session['diagnosis_result'] = result_data

    # 클라이언트에서 `test_result.html`로 이동할 수 있도록 JSON 응답 반환
    return jsonify({"status": "success", "redirect_url": url_for("test_result")})

@app.route('/version', methods=['POST'])
def request_version():
    return jsonify({"resultCode": "0000", "resultData": sw_version})

def webStart():
    """서버 시작"""
    logger.info('Starting dementia web server...')
    opt = dict(host='0.0.0.0', port=configure.port)

    if configure.threadcount:
        opt['threads'] = configure.threadcount

    serve(app, **opt)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=configure.port)
