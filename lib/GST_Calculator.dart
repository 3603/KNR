import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

class GSTCalculator extends StatefulWidget {
  const GSTCalculator({Key? key}) : super(key: key);

  @override
  State<GSTCalculator> createState() => _GSTCalculatorState();
}

class _GSTCalculatorState extends State<GSTCalculator> {
  final _formKey = GlobalKey<FormState>();
  final _amountController = TextEditingController();
  double _gstRate = 18.0; // Default GST rate
  bool _isInclusiveOfGST = false;

  Map<String, double> _results = {
    'baseAmount': 0.0,
    'gstAmount': 0.0,
    'totalAmount': 0.0,
    'cgst': 0.0,
    'sgst': 0.0,
    'igst': 0.0,
  };

  List<double> _gstRates = [0.0, 0.25, 3.0, 5.0, 12.0, 18.0, 28.0];
  bool _isIGST = false; // Toggle between CGST+SGST and IGST

  @override
  void dispose() {
    _amountController.dispose();
    super.dispose();
  }

  void _calculateGST() {
    if (_formKey.currentState!.validate()) {
      double amount = double.parse(_amountController.text);
      double gstRate = _gstRate / 100;

      double baseAmount;
      double gstAmount;
      double totalAmount;

      if (_isInclusiveOfGST) {
        // If amount is inclusive of GST
        baseAmount = amount / (1 + gstRate);
        gstAmount = amount - baseAmount;
        totalAmount = amount;
      } else {
        // If amount is exclusive of GST
        baseAmount = amount;
        gstAmount = amount * gstRate;
        totalAmount = amount + gstAmount;
      }

      double cgst = _isIGST ? 0.0 : gstAmount / 2;
      double sgst = _isIGST ? 0.0 : gstAmount / 2;
      double igst = _isIGST ? gstAmount : 0.0;

      setState(() {
        _results = {
          'baseAmount': baseAmount,
          'gstAmount': gstAmount,
          'totalAmount': totalAmount,
          'cgst': cgst,
          'sgst': sgst,
          'igst': igst,
        };
      });
    }
  }

  String _formatCurrency(double amount) {
    return '₹${amount.toStringAsFixed(2)}';
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('GST Calculator'),
        elevation: 2,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Form(
          key: _formKey,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              // Input card
              Card(
                elevation: 4,
                child: Padding(
                  padding: const EdgeInsets.all(16.0),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text(
                        'Enter Details',
                        style: TextStyle(
                          fontSize: 18,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      const SizedBox(height: 16),

                      // Amount input
                      TextFormField(
                        controller: _amountController,
                        decoration: const InputDecoration(
                          labelText: 'Amount',
                          prefixText: '₹',
                          border: OutlineInputBorder(),
                        ),
                        keyboardType:
                            TextInputType.numberWithOptions(decimal: true),
                        inputFormatters: [
                          FilteringTextInputFormatter.allow(
                              RegExp(r'^\d+\.?\d{0,2}')),
                        ],
                        validator: (value) {
                          if (value == null || value.isEmpty) {
                            return 'Please enter an amount';
                          }
                          if (double.tryParse(value) == null) {
                            return 'Please enter a valid number';
                          }
                          return null;
                        },
                      ),
                      const SizedBox(height: 16),

                      // GST Rate selection
                      DropdownButtonFormField<double>(
                        decoration: const InputDecoration(
                          labelText: 'GST Rate',
                          border: OutlineInputBorder(),
                        ),
                        value: _gstRate,
                        items: _gstRates.map((rate) {
                          return DropdownMenuItem<double>(
                            value: rate,
                            child: Text('${rate.toString()}%'),
                          );
                        }).toList(),
                        onChanged: (value) {
                          setState(() {
                            _gstRate = value!;
                          });
                        },
                      ),
                      const SizedBox(height: 16),

                      // IGST or CGST+SGST Switch
                      Row(
                        children: [
                          const Text('IGST'),
                          Switch(
                            value: _isIGST,
                            onChanged: (value) {
                              setState(() {
                                _isIGST = value;
                              });
                            },
                          ),
                          Text(_isIGST
                              ? 'Inter-state supply'
                              : 'Intra-state supply'),
                        ],
                      ),

                      // Inclusive/Exclusive Switch
                      Row(
                        children: [
                          const Text('Price includes GST'),
                          Switch(
                            value: _isInclusiveOfGST,
                            onChanged: (value) {
                              setState(() {
                                _isInclusiveOfGST = value;
                              });
                            },
                          ),
                        ],
                      ),
                      const SizedBox(height: 16),

                      // Calculate Button
                      SizedBox(
                        width: double.infinity,
                        child: ElevatedButton(
                          onPressed: _calculateGST,
                          style: ElevatedButton.styleFrom(
                            padding: const EdgeInsets.all(12),
                          ),
                          child: const Text(
                            'Calculate',
                            style: TextStyle(fontSize: 16),
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ),

              const SizedBox(height: 24),

              // Results section
              Card(
                elevation: 4,
                child: Padding(
                  padding: const EdgeInsets.all(16.0),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text(
                        'Results',
                        style: TextStyle(
                          fontSize: 18,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      const SizedBox(height: 16),

                      // Result items
                      ResultItem(
                        title: 'Base Amount',
                        value: _formatCurrency(_results['baseAmount']!),
                      ),
                      const Divider(),

                      if (!_isIGST) ...[
                        ResultItem(
                          title: 'CGST (${(_gstRate / 2).toString()}%)',
                          value: _formatCurrency(_results['cgst']!),
                        ),
                        const Divider(),
                        ResultItem(
                          title: 'SGST (${(_gstRate / 2).toString()}%)',
                          value: _formatCurrency(_results['sgst']!),
                        ),
                      ] else ...[
                        ResultItem(
                          title: 'IGST ($_gstRate%)',
                          value: _formatCurrency(_results['igst']!),
                        ),
                      ],

                      const Divider(thickness: 2),
                      ResultItem(
                        title: 'Total GST',
                        value: _formatCurrency(_results['gstAmount']!),
                        isHighlighted: true,
                      ),
                      const Divider(thickness: 2),
                      ResultItem(
                        title: 'Final Amount',
                        value: _formatCurrency(_results['totalAmount']!),
                        isHighlighted: true,
                      ),
                    ],
                  ),
                ),
              ),

              const SizedBox(height: 16),

              // Information Card
              Card(
                elevation: 2,
                color: const Color.fromARGB(255, 19, 20, 21),
                child: Padding(
                  padding: const EdgeInsets.all(16.0),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: const [
                      Text(
                        'GST Information',
                        style: TextStyle(
                          fontSize: 16,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      SizedBox(height: 8),
                      Text(
                        '• CGST: Central Goods and Services Tax\n'
                        '• SGST: State Goods and Services Tax\n'
                        '• IGST: Integrated Goods and Services Tax\n\n'
                        'For intra-state transactions, both CGST and SGST apply.\n'
                        'For inter-state transactions, IGST applies.',
                        style: TextStyle(fontSize: 14),
                      ),
                    ],
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class ResultItem extends StatelessWidget {
  final String title;
  final String value;
  final bool isHighlighted;

  const ResultItem({
    Key? key,
    required this.title,
    required this.value,
    this.isHighlighted = false,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8.0),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(
            title,
            style: TextStyle(
              fontSize: isHighlighted ? 16 : 14,
              fontWeight: isHighlighted ? FontWeight.bold : FontWeight.normal,
            ),
          ),
          Text(
            value,
            style: TextStyle(
              fontSize: isHighlighted ? 16 : 14,
              fontWeight: isHighlighted ? FontWeight.bold : FontWeight.normal,
            ),
          ),
        ],
      ),
    );
  }
}
