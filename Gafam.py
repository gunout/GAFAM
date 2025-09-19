import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class GAFAMStockAnalysis:
    def __init__(self):
        # Définition des GAFAM avec leurs dates d'introduction en bourse
        self.gafam_stocks = {
            'AAPL': {'name': 'Apple Inc.', 'ipo_date': '1980-12-12', 'sector': 'Technology'},
            'MSFT': {'name': 'Microsoft Corp.', 'ipo_date': '1986-03-13', 'sector': 'Technology'},
            'AMZN': {'name': 'Amazon.com Inc.', 'ipo_date': '1997-05-15', 'sector': 'Consumer Cyclical'},
            'GOOGL': {'name': 'Alphabet Inc. (Google)', 'ipo_date': '2004-08-19', 'sector': 'Communication Services'},
            'META': {'name': 'Meta Platforms (Facebook)', 'ipo_date': '2012-05-18', 'sector': 'Communication Services'}
        }
        
        # Événements majeurs affectant les GAFAM
        self.major_events = {
            '2000-03-10': 'Éclatement de la bulle Internet',
            '2008-09-15': 'Crise financière mondiale (Lehman Brothers)',
            '2012-05-18': 'Introduction en bourse de Facebook',
            '2018-03-19': 'Scandale Cambridge Analytica (Facebook)',
            '2018-07-18': 'Amende record de 5Mds$ à Google (UE)',
            '2020-03-23': 'COVID-19: Marchés au plus bas',
            '2020-08-19': 'Apple première entreprise à 2000Mds$',
            '2021-04-28': 'Résultats record Amazon pendant la pandémie',
            '2021-10-28': 'Facebook devient Meta',
            '2022-01-03': 'Correction tech suite aux hausses de taux',
            '2022-10-27': 'Chute de Meta après résultats décevants',
            '2023-01-31': 'Licenciements massifs dans la tech',
            '2023-07-11': 'Adoption de l\'IA générative booste les cours',
            '2024-01-22': 'Nouveaux records pour les géants tech',
            '2024-12-31': 'Projection 2024 (estimation)',
            '2025-12-31': 'Projection 2025 (estimation)'
        }
        
        # Période d'analyse (depuis l'IPO de chaque entreprise jusqu'à 2025)
        self.end_date = '2025-12-31'
        
    def fetch_stock_data(self):
        """Récupère les données boursières pour toutes les actions GAFAM depuis leur IPO"""
        print("📊 Récupération des données boursières des GAFAM depuis leur création...")
        
        all_data = []
        
        for ticker, info in self.gafam_stocks.items():
            print(f"📈 Téléchargement des données pour {info['name']} ({ticker}) depuis {info['ipo_date']}...")
            
            try:
                # Téléchargement des données depuis l'IPO
                stock_data = yf.download(ticker, start=info['ipo_date'], end=self.end_date, progress=False)
                
                if stock_data.empty:
                    print(f"❌ Aucune donnée trouvée pour {ticker}")
                    continue
                
                # Gérer les colonnes MultiIndex si présentes
                if isinstance(stock_data.columns, pd.MultiIndex):
                    stock_data.columns = stock_data.columns.get_level_values(0)
                
                # Vérifier les colonnes disponibles
                print(f"Colonnes disponibles pour {ticker}: {stock_data.columns.tolist()}")
                
                # Si 'Adj Close' n'existe pas, utiliser 'Close' à la place
                if 'Adj Close' not in stock_data.columns:
                    if 'Close' in stock_data.columns:
                        stock_data['Adj Close'] = stock_data['Close']
                        print(f"⚠️ Utilisation de 'Close' à la place de 'Adj Close' pour {ticker}")
                    else:
                        print(f"❌ Ni 'Adj Close' ni 'Close' disponibles pour {ticker}")
                        continue
                
                # Ajouter des colonnes d'information
                stock_data['Ticker'] = ticker
                stock_data['Company'] = info['name']
                stock_data['Sector'] = info['sector']
                stock_data['IPO_Date'] = info['ipo_date']
                
                # Calculer les rendements
                stock_data['Daily_Return'] = stock_data['Adj Close'].pct_change()
                stock_data['Cumulative_Return'] = (1 + stock_data['Daily_Return']).cumprod()
                
                # Calculer les indicateurs techniques
                stock_data['MA_50'] = stock_data['Adj Close'].rolling(window=50).mean()
                stock_data['MA_200'] = stock_data['Adj Close'].rolling(window=200).mean()
                stock_data['Volatility'] = stock_data['Daily_Return'].rolling(window=50).std() * np.sqrt(252)
                
                all_data.append(stock_data)
                
                # Pause pour éviter de surcharger l'API
                import time
                time.sleep(1)
                
            except Exception as e:
                print(f"❌ Erreur lors du traitement de {ticker}: {str(e)}")
                continue
        
        # Combiner toutes les données
        if all_data:
            combined_data = pd.concat(all_data)
            return combined_data
        else:
            return pd.DataFrame()
    
    def simulate_future_data(self, historical_data):
        """Simule les données futures jusqu'en 2025 basées sur les tendances historiques"""
        print("🔮 Simulation des données jusqu'en 2025...")
        
        # Dernière date de données disponibles
        last_date = historical_data.index.max()
        
        # Créer des dates futures jusqu'à fin 2025
        future_dates = pd.date_range(start=last_date + timedelta(days=1), end=self.end_date, freq='D')
        
        future_data = []
        
        for ticker in historical_data['Ticker'].unique():
            # Données historiques pour ce ticker
            ticker_data = historical_data[historical_data['Ticker'] == ticker]
            last_price = ticker_data['Adj Close'].iloc[-1]
            avg_return = ticker_data['Daily_Return'].mean()
            avg_volatility = ticker_data['Daily_Return'].std()
            
            # Simulation de prix futurs (modèle simple avec tendance + volatilité)
            np.random.seed(42)  # Pour la reproductibilité
            n_days = len(future_dates)
            
            # Tendance annuelle moyenne (8% pour les tech)
            daily_trend = (1.08) ** (1/252) - 1
            
            # Générer des rendements aléatoires
            random_returns = np.random.normal(daily_trend, avg_volatility, n_days)
            
            # Calculer les prix futurs
            future_prices = [last_price]
            for ret in random_returns:
                future_prices.append(future_prices[-1] * (1 + ret))
            
            future_prices = future_prices[1:]  # Supprimer le prix initial
            
            # Créer un DataFrame pour les données futures
            future_df = pd.DataFrame({
                'Adj Close': future_prices,
                'Open': future_prices * (1 + np.random.normal(0, 0.01, n_days)),
                'High': future_prices * (1 + np.abs(np.random.normal(0, 0.015, n_days))),
                'Low': future_prices * (1 - np.abs(np.random.normal(0, 0.015, n_days))),
                'Close': future_prices,
                'Volume': np.random.lognormal(15, 1, n_days),  # Volume aléatoire
                'Ticker': ticker,
                'Company': self.gafam_stocks[ticker]['name'],
                'Sector': self.gafam_stocks[ticker]['sector'],
                'IPO_Date': self.gafam_stocks[ticker]['ipo_date'],
                'Daily_Return': random_returns,
                'Is_Forecast': True  # Marquer comme données prévisionnelles
            }, index=future_dates)
            
            # Calculer le rendement cumulatif
            future_df['Cumulative_Return'] = (1 + future_df['Daily_Return']).cumprod() * ticker_data['Cumulative_Return'].iloc[-1]
            
            future_data.append(future_df)
        
        # Combiner avec les données historiques
        if future_data:
            # Réinitialiser l'index pour les données historiques
            historical_reset = historical_data.copy()
            historical_reset['Is_Forecast'] = False
            
            # Combiner
            combined_future = pd.concat(future_data)
            full_data = pd.concat([historical_reset, combined_future], ignore_index=False)
            
            return full_data
        else:
            return historical_data
    
    def calculate_performance_metrics(self, data):
        """Calcule les métriques de performance pour chaque action"""
        print("📐 Calcul des métriques de performance...")
        
        metrics = []
        
        for ticker in data['Ticker'].unique():
            ticker_data = data[data['Ticker'] == ticker]
            
            # Filtrer les données historiques (exclure les prévisions)
            historical_data = ticker_data[~ticker_data.get('Is_Forecast', False)]
            
            if historical_data.empty:
                continue
                
            # Calculs de base
            initial_price = historical_data['Adj Close'].iloc[0]
            final_price = historical_data['Adj Close'].iloc[-1]
            total_return = (final_price / initial_price - 1) * 100
            
            # Calcul du rendement annualisé
            years = (historical_data.index[-1] - historical_data.index[0]).days / 365.25
            annualized_return = ((final_price / initial_price) ** (1/years) - 1) * 100
            
            # Volatilité annualisée
            annual_volatility = historical_data['Daily_Return'].std() * np.sqrt(252) * 100
            
            # Ratio de Sharpe (sans risque à 0 pour simplification)
            sharpe_ratio = annualized_return / annual_volatility if annual_volatility > 0 else 0
            
            # Maximum Drawdown
            cumulative_max = historical_data['Adj Close'].cummax()
            drawdown = (historical_data['Adj Close'] - cumulative_max) / cumulative_max
            max_drawdown = drawdown.min() * 100
            
            # Correlation avec les autres GAFAM
            correlations = {}
            for other_ticker in self.gafam_stocks.keys():
                if other_ticker != ticker:
                    other_data = data[data['Ticker'] == other_ticker]
                    # Aligner les dates
                    aligned_data = historical_data.merge(other_data['Adj Close'], left_index=True, right_index=True, how='inner', suffixes=('', '_other'))
                    if not aligned_data.empty:
                        corr = aligned_data['Adj Close'].corr(aligned_data['Adj Close_other'])
                        correlations[other_ticker] = corr
            
            metrics.append({
                'Ticker': ticker,
                'Company': self.gafam_stocks[ticker]['name'],
                'IPO_Date': self.gafam_stocks[ticker]['ipo_date'],
                'Initial_Price': initial_price,
                'Final_Price': final_price,
                'Total_Return_%': total_return,
                'Annualized_Return_%': annualized_return,
                'Annual_Volatility_%': annual_volatility,
                'Sharpe_Ratio': sharpe_ratio,
                'Max_Drawdown_%': max_drawdown,
                **{f'Corr_with_{k}': v for k, v in correlations.items()}
            })
        
        return pd.DataFrame(metrics)
    
    def create_comprehensive_visualization(self, data, metrics):
        """Crée des visualisations complètes pour l'analyse des GAFAM"""
        print("🎨 Création des visualisations...")
        
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 25))
        
        # Définir la disposition des graphiques
        gs = fig.add_gridspec(5, 2)
        
        # 1. Performance cumulative depuis l'IPO
        ax1 = fig.add_subplot(gs[0, :])
        for ticker in self.gafam_stocks.keys():
            ticker_data = data[data['Ticker'] == ticker]
            ax1.plot(ticker_data.index, ticker_data['Cumulative_Return'], 
                    label=f"{ticker} ({self.gafam_stocks[ticker]['name']})", linewidth=2)
        
        # Ajouter des lignes pour les événements majeurs
        for date_str, event in self.major_events.items():
            try:
                date = pd.to_datetime(date_str)
                if date >= data.index.min() and date <= data.index.max():
                    ax1.axvline(x=date, color='gray', linestyle='--', alpha=0.7)
                    ax1.text(date, ax1.get_ylim()[1] * 0.9, event, rotation=90, verticalalignment='top', 
                            fontsize=8, alpha=0.7)
            except:
                continue
        
        ax1.set_title('Performance Cumulative des GAFAM depuis leur IPO (Base 100)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Rendement Cumulative (Log Scale)')
        ax1.set_yscale('log')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 2. Volatilité annualisée (dernières 50 jours)
        ax2 = fig.add_subplot(gs[1, 0])
        for ticker in self.gafam_stocks.keys():
            ticker_data = data[data['Ticker'] == ticker]
            # Prendre seulement les 1000 derniers points pour la lisibilité
            recent_data = ticker_data.iloc[-1000:] if len(ticker_data) > 1000 else ticker_data
            ax2.plot(recent_data.index, recent_data['Volatility'] * 100, 
                    label=ticker, linewidth=1.5, alpha=0.8)
        
        ax2.set_title('Volatilité Annualisée (50 jours glissants)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Volatilité (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Ratio de Sharpe par entreprise
        ax3 = fig.add_subplot(gs[1, 1])
        sharpe_data = metrics[['Ticker', 'Sharpe_Ratio']].sort_values('Sharpe_Ratio', ascending=False)
        bars = ax3.bar(sharpe_data['Ticker'], sharpe_data['Sharpe_Ratio'])
        ax3.set_title('Ratio de Sharpe (Rendement/Risque)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Ratio de Sharpe')
        
        # Ajouter les valeurs sur les barres
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.2f}', ha='center', va='bottom')
        
        # 4. Heatmap de corrélation
        ax4 = fig.add_subplot(gs[2, 0])
        correlation_data = metrics.filter(regex='Corr_with_')
        correlation_data.columns = [col.replace('Corr_with_', '') for col in correlation_data.columns]
        
        # Créer une matrice de corrélation vide
        correlation_matrix = pd.DataFrame(index=metrics['Ticker'])
        
        for ticker in metrics['Ticker']:
            row = {}
            for other_ticker in metrics['Ticker']:
                if ticker == other_ticker:
                    row[other_ticker] = 1.0
                else:
                    col_name = f'Corr_with_{other_ticker}'
                    if col_name in metrics.columns:
                        row[other_ticker] = metrics[metrics['Ticker'] == ticker][col_name].values[0]
                    else:
                        row[other_ticker] = np.nan
            # Utiliser pd.concat au lieu de append
            correlation_matrix = pd.concat([correlation_matrix, pd.DataFrame([row], index=[ticker])])
        
        sns.heatmap(correlation_matrix.astype(float), annot=True, cmap='coolwarm', center=0, 
                   ax=ax4, square=True, cbar_kws={'shrink': 0.8})
        ax4.set_title('Matrice de Corrélation entre les GAFAM', fontsize=12, fontweight='bold')
        
        # 5. Drawdowns par entreprise
        ax5 = fig.add_subplot(gs[2, 1])
        for ticker in self.gafam_stocks.keys():
            ticker_data = data[data['Ticker'] == ticker]
            cumulative_max = ticker_data['Adj Close'].cummax()
            drawdown = (ticker_data['Adj Close'] - cumulative_max) / cumulative_max * 100
            ax5.plot(ticker_data.index, drawdown, label=ticker, linewidth=1.5, alpha=0.8)
        
        ax5.set_title('Drawdowns Maximums par Entreprise', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Drawdown (%)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Performance annualisée par année
        ax6 = fig.add_subplot(gs[3, :])
        yearly_returns = []
        
        for ticker in self.gafam_stocks.keys():
            ticker_data = data[data['Ticker'] == ticker]
            # Exclure les données prévisionnelles pour les calculs annuels
            historical_data = ticker_data[~ticker_data.get('Is_Forecast', False)]
            
            if not historical_data.empty:
                yearly = historical_data['Adj Close'].resample('Y').last().pct_change().dropna() * 100
                for year, ret in yearly.items():
                    yearly_returns.append({
                        'Year': year.year,
                        'Ticker': ticker,
                        'Return_%': ret
                    })
        
        yearly_df = pd.DataFrame(yearly_returns)
        if not yearly_df.empty:
            yearly_pivot = yearly_df.pivot(index='Year', columns='Ticker', values='Return_%')
            yearly_pivot.plot(kind='bar', ax=ax6)
            ax6.set_title('Rendements Annuels par Entreprise', fontsize=12, fontweight='bold')
            ax6.set_ylabel('Rendement (%)')
            ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax6.grid(True, alpha=0.3)
        
        # 7. Projection jusqu'en 2025
        ax7 = fig.add_subplot(gs[4, :])
        for ticker in self.gafam_stocks.keys():
            ticker_data = data[data['Ticker'] == ticker]
            # Séparer données historiques et prévisionnelles
            historical = ticker_data[~ticker_data.get('Is_Forecast', False)]
            forecast = ticker_data[ticker_data.get('Is_Forecast', False)]
            
            if not historical.empty and not forecast.empty:
                ax7.plot(historical.index, historical['Adj Close'], label=f"{ticker} (Historique)", linewidth=2)
                ax7.plot(forecast.index, forecast['Adj Close'], label=f"{ticker} (Projection)", linewidth=2, linestyle='--')
        
        ax7.set_title('Projection des Prix jusqu\'en 2025', fontsize=12, fontweight='bold')
        ax7.set_ylabel('Prix Ajusté ($)')
        ax7.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax7.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('gafam_stock_analysis_ipo_2025.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def create_individual_company_report(self, data, ticker):
        """Crée un rapport détaillé pour une entreprise spécifique"""
        if ticker not in self.gafam_stocks:
            print(f"❌ {ticker} n'est pas une action GAFAM valide")
            return
        
        print(f"\n📋 Rapport détaillé pour {self.gafam_stocks[ticker]['name']} ({ticker})")
        print("=" * 70)
        
        # Extraire les données de l'entreprise
        company_data = data[data['Ticker'] == ticker]
        
        if company_data.empty:
            print("❌ Aucune donnée disponible pour cette entreprise")
            return
        
        # Données historiques (exclure les prévisions)
        historical_data = company_data[~company_data.get('Is_Forecast', False)]
        
        # Calculs spécifiques
        initial_price = historical_data['Adj Close'].iloc[0]
        current_price = historical_data['Adj Close'].iloc[-1]
        total_return = (current_price / initial_price - 1) * 100
        
        # Calcul du rendement annualisé
        years = (historical_data.index[-1] - historical_data.index[0]).days / 365.25
        annualized_return = ((current_price / initial_price) ** (1/years) - 1) * 100
        
        # Volatilité annualisée
        annual_volatility = historical_data['Daily_Return'].std() * np.sqrt(252) * 100
        
        # Maximum Drawdown
        cumulative_max = historical_data['Adj Close'].cummax()
        drawdown = (historical_data['Adj Close'] - cumulative_max) / cumulative_max
        max_drawdown = drawdown.min() * 100
        
        # Pire performance annuelle
        yearly_returns = historical_data['Adj Close'].resample('Y').last().pct_change().dropna() * 100
        worst_year = yearly_returns.min()
        best_year = yearly_returns.max()
        
        print(f"Date d'IPO: {self.gafam_stocks[ticker]['ipo_date']}")
        print(f"Secteur: {self.gafam_stocks[ticker]['sector']}")
        print(f"Prix initial: ${initial_price:.2f}")
        print(f"Prix actuel: ${current_price:.2f}")
        print(f"Rendement total: {total_return:.2f}%")
        print(f"Rendement annualisé: {annualized_return:.2f}%")
        print(f"Volatilité annualisée: {annual_volatility:.2f}%")
        print(f"Pire drawdown: {max_drawdown:.2f}%")
        print(f"Meilleure année: {best_year:.2f}%")
        print(f"Pire année: {worst_year:.2f}%")
        
        # Visualisations spécifiques à l'entreprise
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Prix et moyennes mobiles
        ax1.plot(historical_data.index, historical_data['Adj Close'], label='Prix Ajusté', linewidth=2)
        ax1.plot(historical_data.index, historical_data['MA_50'], label='Moyenne Mobile 50j', linewidth=1)
        ax1.plot(historical_data.index, historical_data['MA_200'], label='Moyenne Mobile 200j', linewidth=1)
        ax1.set_title(f'{ticker} - Prix et Moyennes Mobiles', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Prix ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Rendements quotidiens
        ax2.hist(historical_data['Daily_Return'].dropna() * 100, bins=100, alpha=0.7, edgecolor='black')
        ax2.set_title(f'{ticker} - Distribution des Rendements Quotidiens', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Rendement (%)')
        ax2.set_ylabel('Fréquence')
        ax2.grid(True, alpha=0.3)
        
        # Ajouter une ligne pour la moyenne
        mean_return = historical_data['Daily_Return'].mean() * 100
        ax2.axvline(mean_return, color='red', linestyle='--', label=f'Moyenne: {mean_return:.2f}%')
        ax2.legend()
        
        # 3. Drawdown
        ax3.plot(historical_data.index, drawdown * 100, label='Drawdown', linewidth=2, color='red')
        ax3.set_title(f'{ticker} - Drawdown Historique', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Drawdown (%)')
        ax3.fill_between(historical_data.index, drawdown * 100, 0, alpha=0.3, color='red')
        ax3.grid(True, alpha=0.3)
        
        # 4. Performance vs S&P 500 (si disponible)
        try:
            # Télécharger les données du S&P 500
            sp500 = yf.download('^GSPC', start=historical_data.index[0], end=historical_data.index[-1], progress=False)
            sp500['Cumulative_Return'] = (1 + sp500['Adj Close'].pct_change()).cumprod()
            
            # Normaliser à 100
            company_norm = historical_data['Cumulative_Return'] * 100
            sp500_norm = sp500['Cumulative_Return'] / sp500['Cumulative_Return'].iloc[0] * 100
            
            ax4.plot(company_norm.index, company_norm, label=ticker, linewidth=2)
            ax4.plot(sp500_norm.index, sp500_norm, label='S&P 500', linewidth=2)
            ax4.set_title(f'{ticker} vs S&P 500 (Performance Relative)', fontsize=12, fontweight='bold')
            ax4.set_ylabel('Performance (Base 100)')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        except:
            # Fallback si échec du téléchargement S&P 500
            yearly_returns = historical_data['Adj Close'].resample('Y').last().pct_change().dropna() * 100
            ax4.bar(yearly_returns.index.year, yearly_returns.values)
            ax4.set_title(f'{ticker} - Rendements Annuels', fontsize=12, fontweight='bold')
            ax4.set_ylabel('Rendement (%)')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{ticker}_detailed_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig

# Fonction principale
def main():
    # Initialiser l'analyseur
    analyzer = GAFAMStockAnalysis()
    
    # Récupérer les données historiques
    stock_data = analyzer.fetch_stock_data()
    
    if stock_data.empty:
        print("❌ Échec de la récupération des données")
        return
    
    # Simuler les données jusqu'en 2025
    full_data = analyzer.simulate_future_data(stock_data)
    
    # Calculer les métriques de performance
    performance_metrics = analyzer.calculate_performance_metrics(full_data)
    
    # Sauvegarder les données
    full_data.to_csv('gafam_stock_data_ipo_2025.csv')
    performance_metrics.to_csv('gafam_performance_metrics.csv', index=False)
    print(f"\n💾 Données sauvegardées dans 'gafam_stock_data_ipo_2025.csv' et 'gafam_performance_metrics.csv'")
    
    # Afficher les métriques de performance
    print("\n📊 Métriques de performance des GAFAM:")
    print("=" * 100)
    print(performance_metrics.round(2).to_string(index=False))
    
    # Créer des visualisations complètes
    analyzer.create_comprehensive_visualization(full_data, performance_metrics)
    
    # Créer des rapports individuels pour chaque entreprise
    for ticker in analyzer.gafam_stocks.keys():
        analyzer.create_individual_company_report(full_data, ticker)
    
    # Analyse comparative
    print("\n🏆 Classement par rendement annualisé:")
    print("=" * 50)
    ranked = performance_metrics.sort_values('Annualized_Return_%', ascending=False)
    for i, (_, row) in enumerate(ranked.iterrows(), 1):
        print(f"{i}. {row['Ticker']}: {row['Annualized_Return_%']:.2f}% (Volatilité: {row['Annual_Volatility_%']:.2f}%)")
    
    print("\n🏆 Classement par ratio de Sharpe:")
    print("=" * 50)
    ranked_sharpe = performance_metrics.sort_values('Sharpe_Ratio', ascending=False)
    for i, (_, row) in enumerate(ranked_sharpe.iterrows(), 1):
        print(f"{i}. {row['Ticker']}: {row['Sharpe_Ratio']:.2f} (Rendement: {row['Annualized_Return_%']:.2f}%)")

if __name__ == "__main__":
    main()