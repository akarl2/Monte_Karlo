import concurrent.futures
import time

if __name__ == '__main__':
    print("hello world")


def function(x, y):
    return x ** y

if __name__ == '__main__':
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        StartTime1 = time.time()

        f1 = executor.submit(function, 3, 4)
        f2 = executor.submit(function, 5, 6)
        f3 = executor.submit(function, 7, 8)

        FinishTime1 = time.time()

        for f in concurrent.futures.as_completed([f1, f2, f3]):
            print("result")

    print(f"multiprocessing took {FinishTime1 - StartTime1}")

    StartTime2 = time.time()

    b1 = function(3, 4)
    b2 = function(5, 6)
    b3 = function(7, 8)

    FinishTime2 = time.time()

    print(f"normal processing took {FinishTime2 - StartTime2}")