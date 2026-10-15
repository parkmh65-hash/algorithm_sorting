from flask import Flask, request, jsonify
import os

app = Flask(__name__)

@app.route('/bubble_sort', methods=['POST'])
def bubble_sort():
    """
    클라이언트로부터 배열을 받아 버블 정렬을 수행하고,
    정렬된 결과와 단계별 시각화를 위한 과정(steps)을 반환합니다.
    """
    data = request.get_json()
    if not data or 'array' not in data:
        return jsonify({'error': 'No array provided'}), 400

    arr = data['array']
    steps = []
    n = len(arr)
    
    # 초기 상태 저장
    steps.append({"array": list(arr), "comparing": [], "swapped": False})
    
    for i in range(n):
        for j in range(0, n - i - 1):
            # 비교 상태 기록
            steps.append({"array": list(arr), "comparing": [j, j+1], "swapped": False})
            
            if arr[j] > arr[j+1]:
                # 값 교환
                arr[j], arr[j+1] = arr[j+1], arr[j]
                # 교환된 상태 기록
                steps.append({"array": list(arr), "comparing": [j, j+1], "swapped": True})
                
    # 최종 정렬 완료 상태 기록
    steps.append({"array": list(arr), "comparing": [], "swapped": False, "finished": True})
    
    return jsonify({
        'sorted_array': arr,
        'steps': steps
    })

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    app.run(debug=True, host='0.0.0.0', port=port)
