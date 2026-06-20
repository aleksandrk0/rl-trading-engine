# tests/check_syntax.py

# 1. Метод _modify_position_stops
success = self._modify_position_stops(
    position=pos,
    new_sl=potential_sl,
    new_tp=pos.tp
)

# 2. Метод _close_partial  
success = self._close_partial(

# 3. process_tick в preprocessor
features = self.preprocessor.process_tick({

# 4. _check_position_exit
self._check_position_exit(tick_data, decision)

# 5. _open_position
self._open_position(

# 6. mock_mt5 убираем везде
tick = self.mock_mt5.symbol_info_tick(self.symbol)

# 7. _model_forward 
outputs[name] = self._model_forward(name, reconstructed)

# 8. _get_tick_data
tick_data = self._get_tick_data()

# 9. Убираем все asyncio.sleep()
time.sleep(1) # вместо await asyncio.sleep(1)

# 10. preprocessor.process_tick
features = self.preprocessor.process_tick(tick_data)

# 11. _get_model_prediction
pred = self._get_model_prediction(model, features)

# 12. _update_trading_state 
self._update_trading_state(predictions, tick_data)

# 13. _process_trading_logic
self._process_trading_logic()

# 14. _display_status
self._display_status()

# 15. cleanup
self.cleanup()

# 16. _predict
pred = self._predict(model, features)

# 17. _calculate_metrics
'metrics': self._calculate_metrics(predictions)

# 18. _handle_signal
self._handle_signal(name, signal, pred)

# 19. _check_position
self._check_position(tick, decision)

# 20. _open_position 
self._open_position(order_type, volume, price)

# 21. _execute_trade
success = self._execute_trade(order_type, volume, price)

# 22. _close_position
self._close_position(position, exit_reason)
