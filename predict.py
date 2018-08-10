import logging, sys

def main():    
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    logging.info('Starting...')
    input_args = get_predict_input_args() 
    
    image_path = input_args.image_path
    checkpoint_path = input_args.checkpoint
    top_k = input_args.top_k
    category_names = input_args.category_names
    
    device ='cpu'    
    if input_args.gpu and torch.cuda.is_available:
        device = 'cuda'     
    
    model, class_from_index = load_checkpoint(checkpoint_path, device)
    model, in_features = get_pretrained_network(arch)    
    model = freeze_layers(model, arch)
    #model, class_from_index = load_checkpoint(model, checkpoint_path, device)
    
    
    
    
if __name__ == "__main__":
    main()