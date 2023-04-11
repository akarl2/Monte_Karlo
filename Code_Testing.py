def test():
    global test_count
    test_count = 0

    def test_1():
        test_count = 25

    test_1()

test()
print(test_count)