---
layout: post
title: C++ main 함수 이해하기
author: minii93
description: main 함수에 대한 설명 및 예제 코드
---

C 또는 C++ 언어로 코딩을 하면 다음과 같이 생긴 main 함수를 정의하게 된다.

```c++
int main(int argc, char *argv[])
```

이 함수는 C/C++ 프로그램을 실행했을 때 호출되는 함수로, 위와 같은 형태를 가지고 있어 프로그램 실행 시에 명령어 인자(command line argument)를 받을 수 있다. `argc`는 argument count의 줄임말로 `argv`가 가리키고 있는 문자열의 개수를 나타내며, `argv`는 argument vector의 줄임말로 문자열 배열을 가리킨다. `argv`의 첫 번째 문자열은 시스템에서 호출한 실행 파일의 이름이며, 프로그램을 실행할 때 추가적인 인자를 전달해주면 그 인자들은 차례대로 두 번째, 세 번째, ... 문자열에 저장된다.

예를 들어, `example.cpp` 파일을 다음과 같이 작성해보자.

```c++
#include <iostream>

int main(int argc, char** argv){
	std::cout << argc << " arguments are passed:" << std::endl;
	for(int i = 0; i < argc; i++){
		std::cout << argv[i] << std::endl;
	}
}
```

터미널에서 컴파일을 해주고

```bash
g++ -c example.cpp
g++ -o example example.o
```

실행하면

```bash
./example Hello World!
```

다음과 같은 출력을 얻는다.

```bash
3 arguments are passed:
./example
Hello
World!
```

위의 실행 코드에서 `./example`이 실행 파일명, `Hello`, `World!`가 각각 파일을 실행할 때 같이 전달해준 명령어 인자에 해당한다.
