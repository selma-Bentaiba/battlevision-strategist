from fpdf import FPDF
import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt
from datetime import datetime

class BattleVisionReport(FPDF):
    def header(self):
        # Logo/Title
        self.set_font('Arial', 'B', 16)
        self.set_text_color(0, 184, 148)  # Green
        self.cell(0, 10, 'BattleVision Strategist - Analysis Report', 0, 1, 'C')
        self.set_text_color(0, 0, 0)
        self.ln(5)
    
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
    
    def chapter_title(self, title):
        self.set_font('Arial', 'B', 14)
        self.set_text_color(0, 184, 148)
        self.cell(0, 10, title, 0, 1, 'L')
        self.set_text_color(0, 0, 0)
        self.ln(2)
    
    def chapter_body(self, body):
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 6, body)
        self.ln()
    
    def add_metric(self, label, value):
        self.set_font('Arial', 'B', 11)
        self.cell(60, 8, label + ':', 0, 0)
        self.set_font('Arial', '', 11)
        self.cell(0, 8, str(value), 0, 1)


def generate_pdf_report(payoff_matrix=None, nash_results=None, simulation_results=None):
    """
    Generate comprehensive PDF report
    """
    pdf = BattleVisionReport()
    pdf.add_page()
    
    # Title and date
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1, 'R')
    pdf.ln(5)
    
    # Executive Summary
    pdf.chapter_title('Executive Summary')
    pdf.chapter_body(
        'This report presents a game-theoretic analysis of adversarial attacks on battlefield '
        'computer vision systems. The analysis covers optimal strategies for both attackers '
        '(using adversarial patches) and defenders (deploying countermeasures).'
    )
    
    # Payoff Matrix Section
    if payoff_matrix is not None:
        pdf.chapter_title('1. Payoff Matrix Analysis')
        
        pdf.chapter_body(
            'The payoff matrix represents the zero-sum game between attackers and defenders. '
            'Positive values indicate attacker advantage, negative values indicate defender advantage.'
        )
        
        # Format matrix as text
        pdf.set_font('Courier', '', 9)
        pdf.cell(0, 6, '                Defense X    Defense Y    Defense Z', 0, 1)
        strategies = ['Patch A (Camo)', 'Patch B (Geo) ', 'Patch C (Text)']
        
        for i, strategy in enumerate(strategies):
            row_text = f'{strategy}:  '
            for j in range(3):
                row_text += f'{payoff_matrix[i, j]:8.3f}    '
            pdf.cell(0, 6, row_text, 0, 1)
        
        pdf.ln(3)
    
    # Nash Equilibrium Section
    if nash_results is not None:
        pdf.chapter_title('2. Nash Equilibrium Analysis')
        
        pdf.set_font('Arial', '', 11)
        pdf.chapter_body(
            'The Nash equilibrium represents the optimal mixed strategies where neither player '
            'can improve their expected payoff by unilaterally changing strategy.'
        )
        
        # Attacker strategy
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 8, 'Optimal Attacker Strategy:', 0, 1)
        pdf.set_font('Arial', '', 11)
        
        strategies_text = [
            f"  - Patch A (Camouflage): {nash_results['attacker'][0]:.1%}",
            f"  - Patch B (Geometric):  {nash_results['attacker'][1]:.1%}",
            f"  - Patch C (Texture):    {nash_results['attacker'][2]:.1%}"
        ]
        for text in strategies_text:
            pdf.cell(0, 6, text, 0, 1)
        
        pdf.ln(3)
        
        # Defender strategy
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 8, 'Optimal Defender Strategy:', 0, 1)
        pdf.set_font('Arial', '', 11)
        
        strategies_text = [
            f"  - Defense X (Denoising):  {nash_results['defender'][0]:.1%}",
            f"  - Defense Y (Ensemble):   {nash_results['defender'][1]:.1%}",
            f"  - Defense Z (Attention):  {nash_results['defender'][2]:.1%}"
        ]
        for text in strategies_text:
            pdf.cell(0, 6, text, 0, 1)
        
        pdf.ln(3)
        
        # Game value
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 8, f'Game Value: {nash_results["value"]:.4f}', 0, 1)
        pdf.set_font('Arial', '', 11)
        
        interpretation = ''
        if nash_results['value'] > 0.1:
            interpretation = (
                'The positive game value indicates an ATTACKER ADVANTAGE at equilibrium. '
                'Even with optimal defense, attackers are expected to succeed more often. '
                'This reflects the asymmetric nature of adversarial warfare.'
            )
        elif nash_results['value'] < -0.1:
            interpretation = (
                'The negative game value indicates a DEFENDER ADVANTAGE at equilibrium. '
                'Advanced defensive measures can effectively counter adversarial patches '
                'when deployed optimally.'
            )
        else:
            interpretation = (
                'The near-zero game value indicates a BALANCED game at equilibrium. '
                'Success depends on execution quality and adaptation speed rather than '
                'inherent strategic advantage.'
            )
        
        pdf.multi_cell(0, 6, interpretation)
        pdf.ln()
    
    # Simulation Results Section
    if simulation_results is not None and simulation_results.get('baseline_detection') is not None:
        pdf.chapter_title('3. Simulation Results')
        
        baseline_count = len(simulation_results.get('baseline_detection', []))
        attacked_count = len(simulation_results.get('attacked_detection', [])) if simulation_results.get('attacked_detection') else 0
        defended_count = len(simulation_results.get('defended_detection', [])) if simulation_results.get('defended_detection') else 0
        
        pdf.add_metric('Baseline Detections', baseline_count)
        pdf.add_metric('Detections After Attack', attacked_count)
        pdf.add_metric('Detections After Defense', defended_count)
        
        attack_success = simulation_results.get('attack_success_rate', 0)
        defense_success = simulation_results.get('defense_success_rate', 0)
        
        pdf.add_metric('Attack Success Rate', f'{attack_success:.1%}')
        pdf.add_metric('Defense Recovery Rate', f'{defense_success:.1%}')
        
        pdf.ln(3)
        
        # Interpretation
        pdf.chapter_body(
            f'The attack reduced detection capability by {attack_success:.1%}, demonstrating '
            f'the vulnerability of AI vision systems to adversarial patches. The defense '
            f'mechanism recovered {defense_success:.1%} of the original detection capability.'
        )
    
    # Strategic Recommendations
    pdf.add_page()
    pdf.chapter_title('4. Strategic Recommendations')
    
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 8, 'For Attackers (Asymmetric Forces):', 0, 1)
    pdf.set_font('Arial', '', 11)
    
    attacker_recommendations = [
        '1. Deploy mixed strategy using multiple patch types to prevent adaptation',
        '2. Focus on cost-effective camouflage and texture patterns ($5-50 range)',
        '3. Randomize patch placement and update frequently',
        '4. Target systems using single detection models',
        '5. Expected success rate: 70-80% against standard defenses'
    ]
    
    for rec in attacker_recommendations:
        pdf.multi_cell(0, 6, rec)
    
    pdf.ln(3)
    
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 8, 'For Defenders (Military/Surveillance):', 0, 1)
    pdf.set_font('Arial', '', 11)
    
    defender_recommendations = [
        '1. Implement ensemble detection models with multiple architectures',
        '2. Deploy attention mechanisms and adaptive filtering',
        '3. Use multi-spectral imaging beyond visible spectrum',
        '4. Combine AI detection with human verification loops',
        '5. Continuously retrain models on adversarial examples',
        '6. Expected recovery rate: 80-90% with advanced defenses'
    ]
    
    for rec in defender_recommendations:
        pdf.multi_cell(0, 6, rec)
    
    pdf.ln(5)
    
    # Key Insights
    pdf.chapter_title('5. Key Insights')
    
    insights = [
        'ASYMMETRIC ADVANTAGE: Low-cost patches ($5-50) can defeat high-value systems ($2M+)',
        '',
        'MIXED STRATEGIES ESSENTIAL: Both sides must randomize to avoid exploitation',
        '',
        'TECHNOLOGY ARMS RACE: Continuous innovation required on both sides',
        '',
        'SCALABILITY: Nash equilibrium strategies apply to networked drone systems',
        '',
        'PRACTICAL IMPLEMENTATION: Real-world effectiveness depends on execution quality'
    ]
    
    for insight in insights:
        if insight:
            pdf.multi_cell(0, 6, 'â€¢ ' + insight)
        else:
            pdf.ln(2)
    
    # Conclusion
    pdf.ln(5)
    pdf.chapter_title('6. Conclusion')
    
    pdf.chapter_body(
        'This analysis demonstrates the critical importance of game theory in modern warfare. '
        'The Nash equilibrium provides optimal strategies for both attackers and defenders, '
        'revealing fundamental asymmetries in adversarial computer vision. '
        '\n\n'
        'Key takeaway: Neither pure strategies nor predictable behavior is viable. Both sides '
        'must employ mixed strategies and continuous adaptation to achieve optimal outcomes '
        'in this zero-sum game.'
    )
    
    # Generate PDF
    pdf_buffer = BytesIO()
    pdf_output = pdf.output(dest='S').encode('latin-1')
    pdf_buffer.write(pdf_output)
    pdf_buffer.seek(0)
    
    return pdf_buffer
