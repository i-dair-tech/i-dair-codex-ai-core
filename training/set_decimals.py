def set_decimals(number):

    number_str = str(number)

    integer_part, decimal_part = number_str.split('.')

    if len(decimal_part) < 3:

        decimal_part = decimal_part.ljust(3, '0')
    else:
        
        decimal_part = decimal_part[:3]


    result = f"{integer_part}.{decimal_part}"

    return float(result)