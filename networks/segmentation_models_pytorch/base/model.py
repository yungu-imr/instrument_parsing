import torch
from . import initialization as init


class SegmentationModelMultiHead(torch.nn.Module):

    def initialize(self):
        init.initialize_decoder(self.decoder_mask)
        init.initialize_head(self.segmentation_head_mask)
        init.initialize_decoder(self.decoder_contour)
        init.initialize_head(self.segmentation_head_contour)
        if self.classification_head is not None:
            init.initialize_head(self.classification_head)

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = self.encoder(x)
        decoder_output_mask = self.decoder_mask(*features)
        decoder_output_contour = self.decoder_contour(*features)

        masks = self.segmentation_head_mask(decoder_output_mask)
        contour = self.segmentation_head_contour(decoder_output_contour)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return masks, contour

    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        with torch.no_grad():
            x = self.forward(x)

        return x


class SegmentationModelMultiHeadPool(torch.nn.Module):

    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)
        # init.initialize_decoder(self.decoder_contour)
        # init.initialize_head(self.segmentation_head_contour)
        if self.classification_head is not None:
            init.initialize_head(self.classification_head)

    def forward(self, x, im):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = self.encoder(x)
        decoder_output_mask = self.decoder(*features)
        masks = self.segmentation_head(decoder_output_mask)
        contour = self.pool_head(masks)
        real_contour = self.real_pool_head(im)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return masks, contour, real_contour

    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        with torch.no_grad():
            x = self.forward(x)

        return x

class SegmentationModel(torch.nn.Module):

    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)
        if self.classification_head is not None:
            init.initialize_head(self.classification_head)

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return masks

    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        with torch.no_grad():
            x = self.forward(x)

        return x
