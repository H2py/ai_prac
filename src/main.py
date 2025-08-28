# main.py

# calculator 모듈을 import
import calculator

print("=== 메인 프로그램 실행 ===")
print(f"main.py의 __name__ 값: {__name__}")
print(f"calculator 모듈의 __name__ 값: {calculator.__name__}")
print()

# calculator 모듈의 함수들 사용
print("Calculator 모듈 함수 사용:")
print(f"20 + 15 = {calculator.add(20, 15)}")
print(f"20 - 15 = {calculator.subtract(20, 15)}")
print(f"20 * 15 = {calculator.multiply(20, 15)}")
print(f"20 / 15 = {calculator.divide(20, 15)}")

print("\n=== 메인 프로그램에서만 실행되는 코드 ===")
if __name__ == "__main__":
    print("이 코드는 main.py가 직접 실행될 때만 보입니다!")
    
    # 간단한 계산기 프로그램
    print("\n간단한 계산기 프로그램:")
    while True:
        try:
            error
            print("\n1. 덧셈")
            print("2. 뺄셈") 
            print("3. 곱셈")
            print("4. 나눗셈")
            print("5. 종료")
            
            choice = input("선택하세요 (1-5): ")
            
            if choice == '5':
                print("프로그램을 종료합니다.")
                break
                
            if choice in ['1', '2', '3', '4']:
                num1 = float(input("첫 번째 숫자: "))
                num2 = float(input("두 번째 숫자: "))
                
                if choice == '1':
                    result = calculator.add(num1, num2)
                    print(f"결과: {num1} + {num2} = {result}")
                elif choice == '2':
                    result = calculator.subtract(num1, num2)
                    print(f"결과: {num1} - {num2} = {result}")
                elif choice == '3':
                    result = calculator.multiply(num1, num2)
                    print(f"결과: {num1} * {num2} = {result}")
                elif choice == '4':
                    result = calculator.divide(num1, num2)
                    print(f"결과: {num1} / {num2} = {result}")
            else:
                print("올바른 선택지를 입력해주세요.")
                
        except ValueError:
            print("올바른 숫자를 입력해주세요.")
        except KeyboardInterrupt:
            print("\n프로그램이 중단되었습니다.")
            break