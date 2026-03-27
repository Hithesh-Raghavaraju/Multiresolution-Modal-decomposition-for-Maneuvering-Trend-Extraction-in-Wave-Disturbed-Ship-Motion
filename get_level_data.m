function full_signal = get_level_data(node, target_level)
   % Функция за извличане на целия сигнал за различно ниво
    % Ако сме на целевото ниво, връщаме неговата реконструкция
    if node.level == target_level
        full_signal = node.reconstruction;
    else
        % Ако има деца, спускаме се надолу и съединяваме резултатите им
        if isfield(node, 'left') && ~isempty(node.left)
            left_part = get_level_data(node.left, target_level);
            right_part = get_level_data(node.right, target_level);
            full_signal = [left_part, right_part];
        else
            % Ако няма повече нива, връщаме нули със същия размер
            full_signal = zeros(size(node.reconstruction));
        end
    end
end