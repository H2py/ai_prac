# calculator.py

def add(a, b):
    """두 수를 더하는 함수"""
    return a + b

def subtract(a, b):
    """두 수를 빼는 함수"""
    return a - b

def multiply(a, b):
    """두 수를 곱하는 함수"""
    return a * b

def divide(a, b):
    """두 수를 나누는 함수"""
    if b == 0:
        return "0으로 나눌 수 없습니다!"
    return a / b

# 모듈이 import될 때는 실행되지 않고,
# 직접 실행될 때만 실행되는 코드
if __name__ == "__main__":
    print("=== Calculator 모듈 테스트 ===")
    print(f"__name__ 값: {__name__}")
    print()
    
    # 테스트 코드
    print("함수 테스트:")
    print(f"10 + 5 = {add(10, 5)}")
    print(f"10 - 5 = {subtract(10, 5)}")
    print(f"10 * 5 = {multiply(10, 5)}")
    print(f"10 / 5 = {divide(10, 5)}")
    print(f"10 / 0 = {divide(10, 0)}")
    
    # 사용자 입력 테스트
    print("\n사용자 입력 테스트:")
    try:
        num1 = float(input("첫 번째 숫자: "))
        num2 = float(input("두 번째 숫자: "))
        operation = input("연산자 (+, -, *, /): ")
        
        if operation == "+":
            result = add(num1, num2)
        elif operation == "-":
            result = subtract(num1, num2)
        elif operation == "*":
            result = multiply(num1, num2)
        elif operation == "/":
            result = divide(num1, num2)
        else:
            result = "잘못된 연산자입니다."
            
        print(f"결과: {result}")
    except KeyboardInterrupt:
        print("\n프로그램이 중단되었습니다.")
    except ValueError:
        print("올바른 숫자를 입력해주세요.")