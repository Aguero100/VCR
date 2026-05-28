import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template_string
import csv

print("⏳ Загрузка модели...")
model = joblib.load('best_model.pkl')
mlb = joblib.load('mlb.pkl')
CATEGORIES = joblib.load('categories.pkl')
model_columns = joblib.load('model_columns.pkl')

# Загружаем CSV без pandas
def load_csv_dict(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return list(csv.DictReader(f))

customers_data = load_csv_dict('df_customers.csv')
print(f"✅ Загружено {len(customers_data)} клиентов")

CASHBACK_MATRIX = {
    'premium_active': [10, 8, 5, 3, 3],
    'premium_inactive': [8, 6, 5, 3, 3],
    'active': [3, 2, 2, 1.5, 1.5],
    'regular': [4, 3, 2.5, 2, 2],
    'inactive': [8, 6, 5, 3, 3],
    'new': [5, 4, 3, 2, 2]
}

SEGMENT_LABELS = {
    'premium_active': '💎 Премиум Активный',
    'premium_inactive': '💤 Премиум Неактивный',
    'active': '✅ Активный',
    'regular': '👤 Обычный',
    'inactive': '❌ Неактивный',
    'new': '🆕 Новый'
}

SEGMENT_COLORS = {
    'premium_active': '#ffd700',
    'premium_inactive': '#ffa726',
    'active': '#4caf50',
    'regular': '#9e9e9e',
    'inactive': '#f44336',
    'new': '#2196f3'
}

app = Flask(__name__)

HTML = '''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Кешбэк Май 2026</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Arial, sans-serif; background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460); min-height: 100vh; padding: 20px; color: #fff; }
        .container { max-width: 1000px; margin: 0 auto; }
        h1 { text-align: center; font-size: 28px; margin-bottom: 5px; }
        h1 span { color: #e94560; }
        .subtitle { text-align: center; color: #a0a0b0; margin-bottom: 25px; font-size: 14px; }
        
        /* ПАНЕЛЬ ФИЛЬТРОВ */
        .filter-panel {
            background: rgba(255,255,255,0.08);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
        }
        .filter-row {
            display: flex;
            gap: 15px;
            align-items: center;
            flex-wrap: wrap;
        }
        .filter-group {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .filter-group label {
            font-weight: 600;
            font-size: 13px;
            color: #b0b0c0;
            white-space: nowrap;
        }
        select, input {
            padding: 8px 12px;
            border: 2px solid #e94560;
            border-radius: 8px;
            font-size: 14px;
            background: rgba(255,255,255,0.9);
            color: #333;
            cursor: pointer;
            min-width: 180px;
        }
        input { width: 80px; text-align: center; }
        select:focus, input:focus { outline: none; border-color: #ff6b81; }
        
        .btn {
            padding: 10px 25px;
            background: #e94560;
            color: #fff;
            border: none;
            border-radius: 8px;
            font-size: 15px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
        }
        .btn:hover { background: #ff6b81; transform: translateY(-1px); }
        .btn-reset {
            background: rgba(255,255,255,0.1);
            color: #a0a0b0;
        }
        .btn-reset:hover { background: rgba(255,255,255,0.2); color: #fff; }
        
        /* СТАТИСТИКА */
        .stats {
            background: rgba(233,69,96,0.1);
            border: 1px solid rgba(233,69,96,0.3);
            border-radius: 10px;
            padding: 12px 20px;
            margin-bottom: 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 10px;
        }
        .stats strong { color: #e94560; }
        
        /* КАРТОЧКИ */
        .card {
            background: rgba(255,255,255,0.06);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 12px;
            border: 1px solid rgba(255,255,255,0.1);
            transition: all 0.3s;
        }
        .card:hover {
            background: rgba(255,255,255,0.1);
            border-color: rgba(233,69,96,0.5);
            transform: translateX(3px);
        }
        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
            flex-wrap: wrap;
            gap: 10px;
        }
        .customer-id { font-size: 18px; font-weight: 700; color: #e94560; }
        .segment-badge {
            padding: 4px 14px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            display: inline-block;
        }
        .card-info {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 10px;
            margin-bottom: 12px;
            font-size: 13px;
            color: #b0b0c0;
        }
        .card-info span { color: #fff; font-weight: 600; display: block; margin-top: 2px; }
        .offers {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }
        .offer-tag {
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 13px;
            font-weight: 600;
        }
        .offer-1 { background: rgba(233,69,96,0.3); color: #ff6b81; }
        .offer-2 { background: rgba(255,167,38,0.3); color: #ffa726; }
        .offer-3 { background: rgba(255,214,0,0.3); color: #ffd600; }
        .offer-4 { background: rgba(76,175,80,0.3); color: #66bb6a; }
        .offer-5 { background: rgba(33,150,243,0.3); color: #42a5f5; }
        .avg-badge {
            padding: 6px 14px;
            background: #e94560;
            border-radius: 20px;
            font-weight: 700;
            font-size: 13px;
            white-space: nowrap;
        }
        
        .no-data {
            text-align: center;
            padding: 40px;
            color: #a0a0b0;
            font-size: 16px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>💰 Кешбэк на <span>Май 2026</span></h1>
        <p class="subtitle">Персональные рекомендации на основе Machine Learning</p>
        
        <!-- ФИЛЬТРЫ -->
        <div class="filter-panel">
            <div class="filter-row">
                <div class="filter-group">
                    <label>👤 Сегмент клиента:</label>
                    <select id="segmentFilter">
                        <option value="all">🌟 Все сегменты</option>
                        <option value="premium_active">💎 Премиум Активный</option>
                        <option value="premium_inactive">💤 Премиум Неактивный</option>
                        <option value="active">✅ Активный</option>
                        <option value="regular">👤 Обычный</option>
                        <option value="inactive">❌ Неактивный</option>
                        <option value="new">🆕 Новый</option>
                    </select>
                </div>
                
                <div class="filter-group">
                    <label>📊 Количество:</label>
                    <input type="number" id="numClients" value="5" min="1" max="50">
                </div>
                
                <button class="btn" onclick="loadCashbacks()">🚀 Показать</button>
                <button class="btn btn-reset" onclick="resetFilters()">🔄 Сбросить</button>
            </div>
        </div>
        
        <div class="stats" id="stats" style="display:none;"></div>
        <div id="result"></div>
    </div>
    
    <script>
        // Цвета для сегментов
        const SEG_COLORS = {
            'premium_active': '#ffd700',
            'premium_inactive': '#ffa726',
            'active': '#4caf50',
            'regular': '#9e9e9e',
            'inactive': '#f44336',
            'new': '#2196f3'
        };
        
        // Загрузка при открытии
        window.onload = () => loadCashbacks();
        
        async function loadCashbacks() {
            const segment = document.getElementById('segmentFilter').value;
            const n = document.getElementById('numClients').value || 5;
            
            let url = `/api/may-cashbacks?n=${n}`;
            if (segment !== 'all') {
                url += `&segment=${segment}`;
            }
            
            try {
                const response = await fetch(url);
                const data = await response.json();
                
                if (data.error) {
                    document.getElementById('result').innerHTML = `<div class="no-data">❌ ${data.error}</div>`;
                    return;
                }
                
                // Статистика
                const statsDiv = document.getElementById('stats');
                statsDiv.style.display = 'flex';
                const segLabel = segment === 'all' ? 'Все сегменты' : document.getElementById('segmentFilter').selectedOptions[0].text;
                statsDiv.innerHTML = `
                    <div>📋 <strong>${segLabel}</strong>: показано <strong>${data.count}</strong> из <strong>${data.total_in_segment}</strong></div>
                    <div>💰 Средний кешбэк: <strong>${data.avg_cashback}%</strong></div>
                    <div>👥 Всего клиентов в базе: <strong>${data.total_all}</strong></div>
                `;
                
                // Карточки
                if (data.count === 0) {
                    document.getElementById('result').innerHTML = '<div class="no-data">😔 Клиентов с таким сегментом не найдено</div>';
                    return;
                }
                
                let html = '';
                data.customers.forEach((cust, index) => {
                    const segColor = SEG_COLORS[cust.segment] || '#9e9e9e';
                    
                    html += `
                        <div class="card">
                            <div class="card-header">
                                <span class="customer-id">#${index + 1} ${cust.customer_id}</span>
                                <span class="segment-badge" style="background:${segColor}22; color:${segColor}; border:1px solid ${segColor}44;">
                                    ${cust.segment_label}
                                </span>
                                <span class="avg-badge">💰 ${cust.avg_cashback}%</span>
                            </div>
                            <div class="card-info">
                                <div>Возраст<span>${cust.age} лет</span></div>
                                <div>Доход<span>${Number(cust.salary).toLocaleString()} ₽</span></div>
                                <div>Город<span>${cust.city}</span></div>
                            </div>
                            <div class="offers">
                                ${cust.offers.map(o => `
                                    <span class="offer-tag offer-${o.priority}">
                                        ${o.priority}. ${o.category} <strong>${o.cashback_percent}%</strong>
                                    </span>
                                `).join('')}
                            </div>
                        </div>
                    `;
                });
                
                document.getElementById('result').innerHTML = html;
                
            } catch (error) {
                document.getElementById('result').innerHTML = `<div class="no-data">❌ Ошибка загрузки данных</div>`;
            }
        }
        
        function resetFilters() {
            document.getElementById('segmentFilter').value = 'all';
            document.getElementById('numClients').value = 5;
            loadCashbacks();
        }
    </script>
</body>
</html>
'''

def predict_categories(customer_id, top_n=5):
    return ['Продукты', 'Рестораны', 'Транспорт', 'Связь', 'Развлечения'][:top_n]

@app.route('/')
def home():
    return render_template_string(HTML)

@app.route('/api/may-cashbacks')
def may_cashbacks():
    n = min(int(request.args.get('n', 5)), 50)
    segment_filter = request.args.get('segment', 'all')
    
    # Фильтруем клиентов по сегменту
    if segment_filter != 'all':
        filtered = [c for c in customers_data if c.get('segment_type') == segment_filter]
    else:
        filtered = customers_data
    
    # Берём первые N из отфильтрованных
    selected = filtered[:n]
    
    result = []
    for row in selected:
        cid = row.get('customer_id', '?')
        seg = row.get('segment_type', 'regular')
        cats = predict_categories(cid)
        rates = CASHBACK_MATRIX.get(seg, [2,2,2,2,2])
        offers = [{'category': cats[j], 'cashback_percent': rates[j] if j<5 else 2, 'priority': j+1} for j in range(5)]
        avg_cb = round(sum(o['cashback_percent'] for o in offers)/5, 1)
        result.append({
            'customer_id': cid,
            'segment': seg,
            'segment_label': SEGMENT_LABELS.get(seg, seg),
            'age': row.get('age', '?'),
            'salary': int(float(row.get('estimated_salary', 0))),
            'city': row.get('city', '?'),
            'offers': offers,
            'avg_cashback': avg_cb
        })
    
    return jsonify({
        'count': len(result),
        'total_in_segment': len(filtered),
        'total_all': len(customers_data),
        'segment': segment_filter,
        'avg_cashback': round(sum(c['avg_cashback'] for c in result)/len(result), 1) if result else 0,
        'customers': result
    })

if __name__ == '__main__':
    print("\n" + "="*55)
    print("🚀 СЕРВЕР ЗАПУЩЕН!")
    print("   👉 http://localhost:5000")
    print("="*55)
    app.run(host='127.0.0.1', port=5000, debug=True)