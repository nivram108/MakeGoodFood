import gpt_2_simple as gpt2
import tensorflow as tf

def load(book):
    tf.reset_default_graph()
    sess = gpt2.start_tf_sess()
    gpt2.load_gpt2(sess, run_name=book)
    return sess
def generate(book, length, temperature, prefix, sess):
    print("generating...")
    single_text = gpt2.generate(sess,
                  run_name=book,
                  length=length,
                  temperature=temperature,
                  prefix=prefix,
                  model_name='355M',
                  nsamples=1,
                  batch_size=1,
                  return_as_list=True
                  )[0]
    single_text = str(single_text)
    print("generated.")
    return get_proper_text(single_text)
def get_proper_text(text):
    text_list = text.split(". ")
    shorten_text = text_list[0] + ". " + text_list[1] + ". "
    if len(shorten_text) < 100 :
        shorten_text = shorten_text + text_list[2]
    return shorten_text
def main():
    book = "pure_review"
    length = 500
    temperature = 0.7
    prefix = input("input prefix: ")
    sess = load(book)
    text_result = generate(book, length, temperature ,prefix, sess)
    print("> " + text_result)
if __name__ == '__main__':
    main()
